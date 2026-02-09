import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score

def calculate_calibration_metrics(y_true, y_prob, n_bins=10):
    """
    Computes Brier Score and Expected Calibration Error (ECE).
    Returns: dict with 'brier' and 'ece'
    """
    brier = brier_score_loss(y_true, y_prob)
    
    # ECE Calculation
    bin_edges = np.linspace(0., 1., n_bins + 1)
    ece = 0.0
    for i in range(len(bin_edges)-1):
        if i == len(bin_edges)-2:
            in_bin = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i+1])
        else:
            in_bin = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i+1])
            
        if np.sum(in_bin) > 0:
            bin_acc = np.mean(y_true[in_bin])
            bin_conf = np.mean(y_prob[in_bin])
            weight = np.sum(in_bin) / len(y_prob)
            ece += np.abs(bin_acc - bin_conf) * weight
            
    return {'brier': brier, 'ece': ece}

def plot_calibration_curve(y_true, prob_dict, save_path=None):
    """
    Plots reliability diagrams for multiple models.
    prob_dict: Dictionary {ModelName: probabilities}
    """
    plt.figure(figsize=(10, 8))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    
    for name, proba in prob_dict.items():
        frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=10)
        ax1.plot(mean_pred, frac_pos, "s-", label=name)
        
    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration Plots (Reliability Curve)')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    plt.show()