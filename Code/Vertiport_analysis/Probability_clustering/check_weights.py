import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def _skew(arr):
    m = arr.mean()
    s = arr.std(ddof=0) + 1e-12
    return np.mean(((arr - m) / s) ** 3)

def _gini(arr):
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0: 
        return np.nan
    if np.sum(arr) == 0:
        return 0.0
    sorted_a = np.sort(arr)
    n = arr.size
    index = np.arange(1, n+1)
    return (2 * np.sum(index * sorted_a) / (n * sorted_a.sum()) - (n + 1) / n)

def check_weights():
    centroid_dir = 'D:/Thesis/UAM/Result/Vertiport_analysis/Probability_clustering/Centroid'
    weight_file = os.path.join(centroid_dir, 'weight_history.npy')
    uam_prob_file = os.path.join(centroid_dir, 'uam_prob_history.npy')
    convergence_file = os.path.join(centroid_dir, 'convergence_history.csv')

    # file existence
    for f in (weight_file, uam_prob_file, convergence_file):
        if not os.path.exists(f):
            print(f"Warning: file not found: {f}")

    # load safely
    try:
        weight_history = np.load(weight_file)
    except Exception as e:
        print("Failed to load weight_history:", e)
        return
    try:
        uam_prob_history = np.load(uam_prob_file)
    except Exception as e:
        print("Failed to load uam_prob_history:", e)
        uam_prob_history = np.array([])

    try:
        convergence_df = pd.read_csv(convergence_file)
    except Exception:
        convergence_df = pd.DataFrame()

    # ensure 2D: (iters, n_weights)
    if weight_history.ndim == 1:
        weight_history = weight_history[np.newaxis, :]
    if uam_prob_history.ndim == 1 and uam_prob_history.size > 0:
        uam_prob_history = uam_prob_history[np.newaxis, :]

    iters = weight_history.shape[0]
    n_weights = weight_history.shape[1]
    print(f"Loaded weight_history shape: {weight_history.shape}")
    print(f"Loaded uam_prob_history shape: {uam_prob_history.shape}")

    # quick checks for bad values
    if not np.isfinite(weight_history).all():
        print("Warning: weight_history contains NaN/Inf")
    if (weight_history < 0).any():
        print("Warning: negative weights found")

    def print_iter_stats(i):
        w = weight_history[i]
        p = np.percentile(w, [0,1,5,10,25,50,75,90,95,99,100])
        print(f"Iteration {i+1}: n={w.size}, min={p[0]:.6f}, 1%={p[1]:.6f}, 5%={p[2]:.6f}, 25%={p[4]:.6f}, "
              f"median={p[5]:.6f}, 75%={p[6]:.6f}, 95%={p[8]:.6f}, max={p[-1]:.6f}")
        print(f"  mean={w.mean():.6f}, std={w.std():.6f}, skew={_skew(w):.3f}, gini={_gini(w):.3f}")
        tot = w.sum() if w.sum() != 0 else 1e-12
        for q in (1,5,10):
            top_share = np.sum(np.sort(w)[-max(1,int(w.size*q/100)):]) / tot
            print(f"  top {q}% weight share: {top_share:.3f}")
        print("")

    # Print first few iteration stats
    for i in range(min(5, iters)):
        print_iter_stats(i)

    # Final iteration diagnostics
    final_weights = weight_history[-1]
    final_uam_probs = uam_prob_history[-1] if uam_prob_history.size > 0 else None

    print("=== FINAL ITERATION ===")
    print_iter_stats(iters - 1)

    # origin/destination split if lengths match expected pattern
    if final_uam_probs is not None and final_weights.size == 2 * final_uam_probs.size:
        n_trips = final_uam_probs.size
        origin_weights = final_weights[:n_trips]
        dest_weights = final_weights[n_trips:]
        print("Origin==Dest (allclose)?", np.allclose(origin_weights, dest_weights))
    else:
        print("Skipping origin/destination comparison (shape mismatch).")

    # correlation between final weights and uam probs (if same length)
    if final_uam_probs is not None and final_uam_probs.size == final_weights.size:
        corr = np.corrcoef(final_weights, final_uam_probs)[0,1]
        print(f"Correlation(weights, uam_probs) = {corr:.3f}")

    # plots (linear + log) and scatter
    plt.figure(figsize=(14,8))
    plt.subplot(2,3,1)
    plt.plot(range(1, iters+1), [np.min(weight_history[i]) for i in range(iters)], label='min')
    plt.plot(range(1, iters+1), [np.max(weight_history[i]) for i in range(iters)], label='max')
    plt.plot(range(1, iters+1), [np.mean(weight_history[i]) for i in range(iters)], label='mean')
    plt.title("Weight stats over iterations"); plt.legend(); plt.grid(alpha=0.3)

    plt.subplot(2,3,2)
    plt.hist(final_weights, bins=50, edgecolor='black')
    plt.title("Final weight distribution (linear)")

    plt.subplot(2,3,3)
    plt.hist(final_weights, bins=50, edgecolor='black', log=True)
    plt.title("Final weight distribution (log y)")

    if final_uam_probs is not None and final_uam_probs.size == final_weights.size:
        plt.subplot(2,3,4)
        plt.scatter(final_uam_probs, final_weights, s=6, alpha=0.6)
        plt.xlabel("UAM prob"); plt.ylabel("Weight")
        plt.title("Weight vs UAM prob (final)")

    plt.tight_layout()
    plt.savefig('D:/Thesis/UAM/Result/Vertiport_analysis/Probability_clustering/Weighting/weight_analysis_improved.png', dpi=200, bbox_inches='tight')
    plt.show()

    # Build CSV with aligned convergence shift (pad/truncate if needed)
    shift_vals = None
    if 'shift' in convergence_df.columns:
        shift_vals = convergence_df['shift'].values
    if shift_vals is None or len(shift_vals) != iters:
        # pad/truncate
        shift_vals = np.full(iters, np.nan)
        if 'shift' in convergence_df.columns:
            L = min(len(shift_vals), len(convergence_df))
            shift_vals[:L] = convergence_df['shift'].values[:L]

    out_df = pd.DataFrame({
        'iteration': np.arange(1, iters+1),
        'weight_min': [np.min(weight_history[i]) for i in range(iters)],
        'weight_max': [np.max(weight_history[i]) for i in range(iters)],
        'weight_mean': [np.mean(weight_history[i]) for i in range(iters)],
        'weight_std': [np.std(weight_history[i]) for i in range(iters)],
        'shift': shift_vals
    })
    out_df.to_csv('D:/Thesis/UAM/Result/Vertiport_analysis/Probability_clustering/Weighting/weight_analysis_detailed.csv', index=False)
    print("Saved detailed CSV and improved plot: weight_analysis_improved.png / weight_analysis_detailed.csv")

if __name__ == "__main__":
    check_weights()
