import numpy as np
import pandas as pd
try:
    import changefinder
except ImportError:
    changefinder = None
    print("Warning: changefinder not installed. Install with: pip install changefinder")


class OutlierDetector:
    """Clase base para detectores de outliers"""
    def detect(self, data, residuals):
        """
        Detecta outliers
        
        Args:
            data: Serie de tiempo original
            residuals: Residuos del modelo
            
        Returns:
            labels: Array de etiquetas ('normal', 'outlier', 'change')
            outlier_score: Score de outliers
            change_score: Score de cambios
        """
        raise NotImplementedError


class DiffDetector(OutlierDetector):
    """
    Detector basado en varianza adaptativa y ChangeFinder.
    Asume que los residuos distribuyen normal.
    """
    def __init__(
        self,
        alpha=0.005,
        quantile=0.995,
        factor_olvido=0.02,
        lag_cambio=2,
        suavizado=7,
        change_quantile=0.99
    ):
        """
        Args:
            alpha: Factor de suavizado para varianza adaptativa
            quantile: Cuantil para threshold de outliers
            factor_olvido: Factor de olvido para ChangeFinder
            lag_cambio: Orden AR para ChangeFinder
            suavizado: Ventana de suavizado para ChangeFinder
            change_quantile: Cuantil para threshold de cambios
        """
        self.alpha = alpha
        self.quantile = quantile
        self.factor_olvido = factor_olvido
        self.lag_cambio = lag_cambio
        self.suavizado = suavizado
        self.change_quantile = change_quantile
        
        if changefinder is None:
            raise ImportError("changefinder is required. Install with: pip install changefinder")
    
    def detect(self, data, residuals):
        """Detecta outliers usando varianza adaptativa y ChangeFinder"""
        n = len(data)
        
        # 1. Calcular varianza adaptativa
        serie_residuos = pd.Series(residuals)
        var_inicial = np.nanvar(serie_residuos.values)
        if not np.isfinite(var_inicial) or var_inicial <= 0:
            var_inicial = 1.0
        
        s2 = []
        s2_curr = var_inicial
        for r in serie_residuos.values:
            if np.isfinite(r):
                s2_curr = (1 - self.alpha) * s2_curr + self.alpha * (r**2)
            s2.append(max(s2_curr, 1e-8))
        s2 = np.array(s2, dtype=float)
        
        # 2. Calcular outlier score (log-likelihood negativa)
        outlier_score = 0.5 * (np.log(2 * np.pi * s2) + (residuals**2) / s2)
        
        # 3. Threshold para outliers
        valid_out = np.isfinite(outlier_score)
        thr_out = np.quantile(outlier_score[valid_out], self.quantile) if valid_out.any() else np.inf
        is_out = outlier_score >= thr_out
        
        # 4. ChangeFinder para detectar cambios
        cf = changefinder.ChangeFinder(
            r=self.factor_olvido, 
            order=self.lag_cambio, 
            smooth=self.suavizado
        )
        score_cambio = np.array([cf.update(float(v)) for v in data], dtype=float)
        
        # 5. Threshold para cambios
        valid_ch = np.isfinite(score_cambio)
        thr_ch = np.quantile(score_cambio[valid_ch], self.change_quantile) if valid_ch.any() else np.inf
        is_ch = score_cambio >= thr_ch
        
        # 6. Generar labels
        labels = np.full(n, "normal", dtype=object)
        labels[is_out] = "outlier"
        labels[is_ch] = "change"
        
        return labels, outlier_score, score_cambio