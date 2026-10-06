"""
@file generate_plots.py
@description CLI script to render and save all publication plots
@module experiments/generate_plots
"""

from src.visualizations import generate_all_plots


def main():
    print("=" * 70)
    print(" GENERATING PUBLICATION VISUALIZATIONS & CONFUSION MATRICES")
    print("=" * 70)
    generate_all_plots()
    print("=" * 70)
    print("[+] All plots successfully generated in 'results/plots/'")
    print("=" * 70)


if __name__ == "__main__":
    main()
