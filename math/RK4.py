import numpy as np
from typing import Callable

def rk4_step(derivatives: Callable, state: np.ndarray, dt: float, t: float) -> np.ndarray:
    """
    Один шаг интегрирования методом Рунге-Кутта 4-го порядка
    
    Args:
        derivatives: функция расчета производных state'(t) = f(state, t)
        state: текущее состояние
        dt: шаг интегрирования
        t: текущее время
        
    Returns:
        np.ndarray: новое состояние
    """
    k1 = derivatives(state, t)
    k2 = derivatives(state + dt/2 * k1, t + dt/2)
    k3 = derivatives(state + dt/2 * k2, t + dt/2)
    k4 = derivatives(state + dt * k3, t + dt)
    
    return state + dt/6 * (k1 + 2*k2 + 2*k3 + k4)