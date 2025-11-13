import numpy as np
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA

class TimeSeriesModel:
    """Clase base para modelos de series de tiempo"""
    def fit(self, data):
        raise NotImplementedError
    
    def get_residuals(self, data):
        raise NotImplementedError


class ARModel(TimeSeriesModel):
    def __init__(self, q=2):  # q porque en tu config usas 'q'
        """
        Modelo Autoregresivo AR
        
        Args:
            q: Orden del modelo AR (lags)
        """
        self.q = q
        self.model = None
        self.fitted_model = None
    
    def fit(self, data):
        """Ajusta el modelo AR a los datos"""
        self.model = AutoReg(data, lags=self.q, old_names=False)
        self.fitted_model = self.model.fit()
        return self
    
    def get_residuals(self, data):
        """Calcula los residuos del modelo"""
        if self.fitted_model is None:
            self.fit(data)
        
        n = len(data)
        pred = self.fitted_model.predict(start=self.q, end=n-1, dynamic=False)
        
        # Crear array de residuos con NaN para los primeros q valores
        residuos = np.full(n, np.nan, dtype=float)
        residuos[self.q:] = data[self.q:] - pred
        
        return residuos


class MAModel(TimeSeriesModel):
    def __init__(self, q=2):
        """
        Modelo de Media Móvil MA(q)
        
        Args:
            q: Orden del modelo MA
        """
        self.q = q
        self.model = None
        self.fitted_model = None
    
    def fit(self, data):
        """Ajusta el modelo MA a los datos"""
        # MA(q) es equivalente a ARIMA(0, 0, q)
        self.model = ARIMA(data, order=(0, 0, self.q))
        self.fitted_model = self.model.fit()
        return self
    
    def get_residuals(self, data):
        """Calcula los residuos del modelo"""
        if self.fitted_model is None:
            self.fit(data)
        
        return self.fitted_model.resid


class ARMAModel(TimeSeriesModel):
    def __init__(self, p=1, q=1):
        """
        Modelo ARMA(p,q)
        
        Args:
            p: Orden AR
            q: Orden MA
        """
        self.p = p
        self.q = q
        self.model = None
        self.fitted_model = None
    
    def fit(self, data):
        """Ajusta el modelo ARMA a los datos"""
        # ARMA(p,q) es equivalente a ARIMA(p, 0, q)
        self.model = ARIMA(data, order=(self.p, 0, self.q))
        self.fitted_model = self.model.fit()
        return self
    
    def get_residuals(self, data):
        """Calcula los residuos del modelo"""
        if self.fitted_model is None:
            self.fit(data)
        
        return self.fitted_model.resid