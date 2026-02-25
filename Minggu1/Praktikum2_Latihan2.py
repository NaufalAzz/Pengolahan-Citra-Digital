import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# Paksa backend supaya grafik muncul di Windows
matplotlib.use('TkAgg')


def simulate_digitization(analog_signal, sampling_rate, quantization_levels):
    """
    analog_signal: fungsi kontinu f(x)
    sampling_rate: jumlah sampel per interval
    quantization_levels: jumlah level kuantisasi
    """

    print("\n=== SIMULASI DIGITALISASI ===")
    print(f"Sampling Rate        : {sampling_rate}")
    print(f"Quantization Levels  : {quantization_levels}")

    # ==========================
    # SINYAL ANALOG (Kontinu)
    # ==========================
    x_continuous = np.linspace(0, 2*np.pi, 1000)
    y_continuous = analog_signal(x_continuous)

    # ==========================
    # SAMPLING
    # ==========================
    x_sampled = np.linspace(0, 2*np.pi, sampling_rate)
    y_sampled = analog_signal(x_sampled)

    # ==========================
    # QUANTIZATION
    # ==========================
    y_min = y_sampled.min()
    y_max = y_sampled.max()

    # Buat level kuantisasi
    q_levels = np.linspace(y_min, y_max, quantization_levels)

    # Mapping ke level terdekat
    indices = np.digitize(y_sampled, q_levels)
    indices = np.clip(indices - 1, 0, quantization_levels - 1)
    y_quantized = q_levels[indices]

    # ==========================
    # VISUALISASI
    # ==========================
    plt.figure(figsize=(10, 6))

    # Sinyal analog (halus)
    plt.plot(x_continuous, y_continuous, label="Analog Signal", linewidth=2)

    # Titik sampling
    plt.stem(x_sampled, y_sampled, basefmt=" ", label="Sampled Signal")

    # Sinyal hasil kuantisasi
    plt.step(x_sampled, y_quantized, where='mid', label="Quantized Signal")

    plt.title("Simulasi Sampling dan Quantization")
    plt.xlabel("x")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)

    plt.show()


# ==========================
# MAIN PROGRAM
# ==========================
if __name__ == "__main__":
    simulate_digitization(
        analog_signal=lambda x: np.sin(x),
        sampling_rate=20,
        quantization_levels=8
    )
