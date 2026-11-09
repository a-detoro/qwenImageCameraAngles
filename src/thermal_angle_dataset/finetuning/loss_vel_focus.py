"""
Loss de Flow Matching para el finetuning del DiT de Qwen-Image, con
ponderacion espacial de la velocity en los tokens donde caen los
keypoints de pose del ground truth.

Diseño:
    No se decodifica nada y no se ejecuta ningun modelo auxiliar (HRNet,
    VAE decoder) dentro del paso de loss. La senal de pose se proyecta
    directamente sobre la rejilla de tokens del transformer mediante un
    resampleo geometrico determinista del heatmap de keypoints, y ese
    mapa de pesos multiplica el error cuadratico de la velocity antes
    del .mean() final.

    Esto funciona porque _pack_latents (en el pipeline de diffusers)
    aplana la rejilla de bloques 2x2 del latente en orden row-major:
    token n = h_block * (W//2) + w_block. El resampleo del heatmap con
    F.interpolate + flatten(2) reproduce exactamente ese mismo orden,
    asi que el peso del token n corresponde, sin desalineacion, a la
    misma posicion espacial que ve el transformer en v_pred[:, n, :].

    El downscale total pixel -> token es vae_scale_factor * patch_size
    (8 * 2 = 16 por defecto en Qwen-Image): un token cubre un bloque de
    16x16 pixeles de la imagen original.

    target_velocity (v = eps - x0) es exacta para cualquier t, asi que
    a diferencia de un x0_pred reconstruido en un solo paso, no hace
    falta desconfiar de ella a t alto. Por eso esta loss NO pondera por
    fiabilidad de reconstruccion.

    use_t_scaling (flag mas abajo) es una hipotesis distinta y SIN
    validar: que a t alto (mucho ruido) el modelo opera sobre estructura
    gruesa/global y a t bajo sobre detalle fino, y que la posicion
    exacta de una articulacion es detalle fino. Esta aislado para poder
    desactivarse con un solo cambio de flag, o borrando el bloque
    indicado, sin tocar el resto de la loss.
"""

import logging
from typing import Optional, Sequence

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# Peso relativo de cada keypoint al colapsar el heatmap a un unico mapa
# espacial. Orden y semantica: formato COCO-pose extendido con manos y
# pies (27 puntos). Editar aqui si cambia el esquema de anotacion.
DEFAULT_KEYPOINT_WEIGHTS = [
    1.0,              # nose
    0.8, 0.8,         # eyes
    0.6, 0.6,         # ears
    1.5, 1.5,         # shoulders
    1.2, 1.2,         # elbows
    1.0, 1.0,         # wrists
    1.5, 1.5,         # hips
    1.2, 1.2,         # knees
    1.0, 1.0,         # ankles
    0.8, 0.8,         # pinky
    0.8, 0.8,         # index
    0.8, 0.8,         # thumb
    0.8, 0.8,         # heel
    0.8, 0.8,         # foot_index
]


class CombinedLossFn:
    """Velocity loss de Flow Matching ponderada espacialmente por donde
    caen los keypoints de pose, proyectados a la rejilla de tokens.

    No usa VAE decoder ni red de pose auxiliar: toda la ponderacion se
    construye con downsampling geometrico del heatmap GT, sin pasar
    gradiente por esa parte del calculo.
    """

    def __init__(
        self,
        img_height: int = 512,
        img_width: int = 512,
        vae_scale_factor: int = 8,
        patch_size: int = 2,
        pose_weight_scale: float = 4.0,
        keypoint_weights: Optional[Sequence[float]] = None,
        # ----------------------------------------------------------------
        # ABLATION: ver docstring del modulo y de _t_scaling_factor.
        # Para desactivar esta hipotesis basta con use_t_scaling=False.
        # Para eliminarla del todo: borrar este flag, el metodo
        # _t_scaling_factor y el bloque marcado en _build_pose_weight.
        use_t_scaling: bool = True,
        # ----------------------------------------------------------------
    ):
        self.img_height = img_height
        self.img_width = img_width
        self.pose_weight_scale = pose_weight_scale
        self.use_t_scaling = use_t_scaling

        # downscale total de pixel a token: VAE (conv con stride) + patchify
        self.token_downscale = vae_scale_factor * patch_size
        self.seq_h = img_height // self.token_downscale
        self.seq_w = img_width // self.token_downscale

        weights = keypoint_weights if keypoint_weights is not None else DEFAULT_KEYPOINT_WEIGHTS
        self.keypoint_weights = torch.tensor(weights, dtype=torch.float32)

        self.current_target_heatmaps: Optional[torch.Tensor] = None
        self.current_timesteps: Optional[torch.Tensor] = None

        self.step_counter = 0

        # valores expuestos para logging/inspeccion externa
        self.last_velocity_loss = 0.0      # MSE sin ponderar, de referencia
        self.last_weighted_loss = 0.0      # lo que de verdad se backpropaga
        self.last_pose_weight_mean = 1.0
        self.last_t_scaling_mean = 1.0

    def set_batch_context(self, target_heatmaps: torch.Tensor, timesteps: torch.Tensor) -> None:
        """Llamar una vez por step de entrenamiento, antes de __call__.

        target_heatmaps: (B, K, Hh, Wh) heatmaps GT en cualquier
            resolucion; se reescalan internamente a (seq_h, seq_w).
        timesteps: (B,) en la misma escala que usa el scheduler (se
            asume /1000 para normalizar a [0,1]). Solo se usan si
            use_t_scaling=True.
        """
        self.current_target_heatmaps = target_heatmaps
        self.current_timesteps = timesteps

    @torch.no_grad()
    def _t_scaling_factor(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Hipotesis a validar por ablation: a t alto (mucho ruido) el
        modelo trabaja sobre estructura gruesa/global; a t bajo, sobre
        detalle fino, donde vive la posicion exacta de una articulacion.
        Si se confirma, tiene sentido que el EXTRA de peso por pose se
        atenue cuando t es alto, en vez de exigir precision de
        articulacion cuando el propio nivel de ruido la hace poco
        recuperable. El piso de peso 1.0 (velocity loss estandar) nunca
        se toca, solo el extra.

        Para desactivar: use_t_scaling=False en el constructor.
        Para borrar del todo: eliminar este metodo y el bloque marcado
        en _build_pose_weight, dejando solo `extra` sin multiplicar.
        """
        if self.current_timesteps is None:
            logger.warning(
                f"[LOSS step {self.step_counter}] use_t_scaling=True pero no "
                f"hay timesteps en contexto; se usa t_scale=1.0 este step."
            )
            return torch.ones(1, 1, 1, device=device, dtype=dtype)

        t_norm = (self.current_timesteps.float() / 1000.0).clamp(0.0, 1.0)
        t_scale = (1.0 - t_norm).view(-1, 1, 1).to(device=device, dtype=dtype)
        self.last_t_scaling_mean = t_scale.mean().item()
        return t_scale

    @torch.no_grad()
    def _build_pose_weight(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Construye el peso por token: (B, seq_len, 1).

        Fondo = 1.0 (la velocity loss estandar, intacta). Encima se suma
        pose_weight_scale * senal_de_pose, donde senal_de_pose es el
        heatmap GT colapsado a un unico canal (maximo ponderado sobre
        keypoints) y reescalado a (seq_h, seq_w) con el mismo orden
        row-major que _pack_latents usa para construir la secuencia.

        Todo bajo no_grad: es una mascara fija derivada de anotaciones,
        no una cantidad aprendida. El gradiente real fluye solo por
        v_pred en __call__.
        """
        seq_len = self.seq_h * self.seq_w

        if self.current_target_heatmaps is None:
            return torch.ones(batch_size, seq_len, 1, device=device, dtype=dtype)

        heatmaps = self.current_target_heatmaps.to(device).float()

        if heatmaps.shape[1] != self.keypoint_weights.numel():
            raise ValueError(
                f"target_heatmaps tiene {heatmaps.shape[1]} canales de keypoint, "
                f"pero keypoint_weights tiene {self.keypoint_weights.numel()}. "
                f"Revisa el esquema de anotacion o pasa keypoint_weights explicito."
            )

        kw = self.keypoint_weights.to(device).view(1, -1, 1, 1)

        # Colapsa K canales de keypoint a 1 mapa espacial: el keypoint mas
        # relevante presente en cada celda domina su peso. Alternativa a
        # probar: .sum(dim=1) en vez de .amax si se prefiere acumular
        # cuando varios keypoints caen cerca en vez de tomar el maximo.
        combined = (heatmaps * kw).amax(dim=1, keepdim=True)  # (B,1,Hh,Wh)

        # mode="area" promedia dentro de cada celda destino: es el modo
        # correcto para downsampling (evita perder picos finos que
        # bilinear/nearest podrian saltarse al reducir tanto la resolucion).
        pose_signal = F.interpolate(combined, size=(self.seq_h, self.seq_w), mode="area")
        # flatten(2) aplana (seq_h, seq_w) con seq_h como eje lento y
        # seq_w como eje rapido: el mismo orden row-major que produce
        # _pack_latents al aplanar bloque-fila/bloque-columna.
        pose_signal = pose_signal.flatten(2).transpose(1, 2)  # (B, seq_len, 1)

        # normaliza a [0,1] por muestra: pose_weight_scale tiene el mismo
        # significado sin importar la intensidad absoluta con la que se
        # hayan renderizado los heatmaps de entrada
        peak = pose_signal.amax(dim=1, keepdim=True).clamp_min(1e-8)
        pose_signal = pose_signal / peak

        extra = self.pose_weight_scale * pose_signal

        # ============================== ABLATION ==============================
        # Bloque aislado a proposito. Para desactivar la hipotesis de t_weight
        # sin tocar nada mas: usar use_t_scaling=False (recomendado), o borrar
        # estas dos lineas y dejar `extra` tal cual.
        if self.use_t_scaling:
            extra = extra * self._t_scaling_factor(device, dtype)
        # ============================ FIN ABLATION =============================

        weight = 1.0 + extra
        self.last_pose_weight_mean = weight.mean().item()
        return weight.to(dtype)

    def __call__(self, outputs: torch.Tensor, combined_target: torch.Tensor) -> torch.Tensor:
        """outputs: v_pred del transformer, (B, seq_len, C).
        combined_target[:, 0]: v_target = eps - x0, misma forma.
        combined_target[:, 1]: noisy_latents; ya no se usa en esta version
            (no hay reconstruccion de x0 ni decode), se mantiene en la
            forma de entrada solo por compatibilidad con el collate/data
            loader existente.
        """
        device = outputs.device

        v_target = combined_target[:, 0, :, :].float()
        v_pred = outputs.float()

        if v_pred.shape[1] > v_target.shape[1]:
            v_pred = v_pred[:, : v_target.shape[1], :]

        if torch.isnan(v_pred).any() or torch.isinf(v_pred).any():
            logger.warning(
                f"[LOSS step {self.step_counter}] NaN/Inf en v_pred. "
                f"max={v_pred.abs().nanmax().item():.2f}"
            )
            self.step_counter += 1
            return torch.tensor(0.0, device=device, requires_grad=True)

        batch_size, seq_len = v_target.shape[0], v_target.shape[1]
        weight = self._build_pose_weight(batch_size, device, v_pred.dtype)

        assert weight.shape[1] == seq_len, (
            f"pose weight tiene {weight.shape[1]} tokens pero v_target tiene "
            f"{seq_len}. Revisa img_height/img_width del constructor contra "
            f"la resolucion real de este batch (configurado a "
            f"{self.img_height}x{self.img_width})."
        )

        # promediar canales antes de ponderar es equivalente a ponderar
        # canal por canal y promediar despues (weight no depende del
        # canal), pero opera sobre un tensor 64x mas pequeno
        sq_err = (v_pred - v_target).pow(2).mean(dim=-1, keepdim=True)  # (B, seq_len, 1)

        velocity_loss_val = sq_err.mean()             # MSE plano, de referencia/logging
        weighted_loss_val = (sq_err * weight).mean()  # esto es lo que se backpropaga

        self.last_velocity_loss = velocity_loss_val.item()
        self.last_weighted_loss = weighted_loss_val.item()

        if self.step_counter % 50 == 0:
            logger.info(
                f"[LOSS step {self.step_counter}] "
                f"velocity(plano): {velocity_loss_val.item():.6f}, "
                f"velocity(pose-weighted): {weighted_loss_val.item():.6f}, "
                f"peso medio: {self.last_pose_weight_mean:.3f}, "
                f"use_t_scaling: {self.use_t_scaling} "
                f"(media t_scale: {self.last_t_scaling_mean:.3f})"
            )

        self.step_counter += 1
        return weighted_loss_val


if __name__ == "__main__":
    # Smoke test minimo: shapes correctas, backward funciona, y se puede
    # alternar use_t_scaling sin tocar nada mas. No sustituye probar contra
    # el training loop real, pero confirma que el archivo es ejecutable.
    logging.basicConfig(level=logging.INFO)

    img_h = img_w = 512
    probe = CombinedLossFn(img_height=img_h, img_width=img_w)
    seq_len = probe.seq_h * probe.seq_w
    C = 64
    B = 2

    v_pred = torch.randn(B, seq_len, C, requires_grad=True)
    v_target = torch.randn(B, seq_len, C)
    noisy_latents_unused = torch.zeros(B, seq_len, C)  # slot 1, vestigial, ver __call__
    combined_target = torch.stack([v_target, noisy_latents_unused], dim=1)

    K = len(DEFAULT_KEYPOINT_WEIGHTS)
    heatmaps = torch.zeros(B, K, 96, 96)
    heatmaps[:, 5, 10:14, 10:14] = 1.0  # simula un keypoint de hombro
    timesteps = torch.tensor([200.0, 800.0])  # un sample con poco ruido, otro con mucho

    for use_t in (True, False):
        loss_fn = CombinedLossFn(img_height=img_h, img_width=img_w, use_t_scaling=use_t)
        loss_fn.set_batch_context(heatmaps, timesteps)
        loss = loss_fn(v_pred, combined_target)
        loss.backward(retain_graph=True)
        print(
            f"use_t_scaling={use_t} | loss={loss.item():.4f} | "
            f"velocity_plano={loss_fn.last_velocity_loss:.4f} | "
            f"peso_medio={loss_fn.last_pose_weight_mean:.3f} | "
            f"grad_norm={v_pred.grad.norm().item():.4f}"
        )
        v_pred.grad = None