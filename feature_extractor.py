import numpy as np
import pywt
from scipy.signal import (
    welch, coherence, butter, sosfiltfilt,
    hilbert, find_peaks
)
from sklearn.cluster import KMeans


# ================= 1. PSD、DE、Rel =================
def calc_psd_de_rel(X_filt, sfreq=500):
    bands = [(4, 8), (8, 13), (13, 30)]
    psd_feats, de_feats, rel_feats = [], [], []
    n_trials, n_channels, n_times = X_filt.shape

    for t in range(n_trials):
        t_psd, t_de, t_rel = [], [], []

        for c in range(n_channels):
            freqs, psd = welch(
                X_filt[t, c, :],
                fs=sfreq,
                nperseg=n_times
            )

            total_power = np.mean(
                psd[(freqs >= 4) & (freqs < 45)]
            )

            for low, high in bands:
                mask = (freqs >= low) & (freqs < high)
                mean_psd = np.mean(psd[mask])

                t_psd.append(mean_psd)
                t_de.append(
                    0.5 * np.log(
                        2 * np.pi * np.e * mean_psd + 1e-12
                    )
                )
                t_rel.append(
                    mean_psd / (total_power + 1e-12)
                )

        psd_feats.append(t_psd)
        de_feats.append(t_de)
        rel_feats.append(t_rel)

    return (
        np.asarray(psd_feats),
        np.asarray(de_feats),
        np.asarray(rel_feats)
    )


# ================= 2. Hjorth =================
def calc_hjorth(X_filt):
    d1 = np.diff(X_filt, axis=2)
    d2 = np.diff(d1, axis=2)

    var_s = np.var(X_filt, axis=2)
    var_d1 = np.var(d1, axis=2)
    var_d2 = np.var(d2, axis=2)

    act = var_s
    mob = np.sqrt(var_d1 / (var_s + 1e-12))
    comp = (
        np.sqrt(var_d2 / (var_d1 + 1e-12))
        / (mob + 1e-12)
    )

    return np.stack(
        [act, mob, comp],
        axis=-1
    ).reshape(X_filt.shape[0], -1)


# ================= 3. Permutation Entropy =================
def calc_perm_entropy(X_filt, order=3):
    n_trials, n_channels, n_times = X_filt.shape
    pe = np.zeros((n_trials, n_channels))

    for t in range(n_trials):
        for c in range(n_channels):
            sig = X_filt[t, c, :].tolist()
            counts = [0] * 6

            for i in range(n_times - order):
                a = sig[i]
                b = sig[i + 1]
                c_val = sig[i + 2]

                if a < b:
                    if b < c_val:
                        idx = 0
                    elif a < c_val:
                        idx = 1
                    else:
                        idx = 2
                else:
                    if a < c_val:
                        idx = 3
                    elif b < c_val:
                        idx = 4
                    else:
                        idx = 5

                counts[idx] += 1

            total = n_times - order
            p = np.array(counts) / total
            p = p[p > 0]

            pe[t, c] = (
                -np.sum(p * np.log(p))
                / np.log(6.0)
            )

    return pe


# ================= 4. Coherence =================
def calc_coherence(X_filt, sfreq=500):
    bands = [(4, 8), (8, 13), (13, 30)]

    ch_names = [
        "FP1", "FP2", "Fz", "F3", "F4", "F7", "F8", "FCz",
        "FC3", "FC4", "FT7", "FT8", "Cz", "C3", "C4", "T3",
        "T4", "CPz", "CP3", "CP4", "TP7", "TP8", "Pz", "P3",
        "P4", "T5", "T6", "Oz", "O1", "O2", "HEOL", "HEOR"
    ]

    selected_pairs = [
        ("FP1", "FP2"),
        ("F3", "F4"),
        ("F7", "F8"),
        ("Fz", "Cz"),
        ("Fz", "Pz"),
        ("F3", "P3"),
        ("F4", "P4"),
        ("C3", "C4"),
        ("P3", "P4"),
        ("O1", "O2")
    ]

    ch_to_idx = {
        name: i for i, name in enumerate(ch_names)
    }

    pair_indices = [
        (ch_to_idx[ch1], ch_to_idx[ch2])
        for ch1, ch2 in selected_pairs
    ]

    n_trials, n_channels, n_times = X_filt.shape
    n_features = len(selected_pairs) * len(bands)

    coherence_features = np.zeros(
        (n_trials, n_features),
        dtype=np.float64
    )

    nperseg = min(256, n_times)

    print(
        f"正在计算 Coherence: "
        f"{n_trials} 个 Trial × "
        f"{len(selected_pairs)} 个通道对 × "
        f"{len(bands)} 个频段"
    )

    print("选用的10对电极：")
    for i, (ch1, ch2) in enumerate(selected_pairs, 1):
        print(f"  {i}. {ch1}-{ch2}")

    for t in range(n_trials):
        trial = X_filt[t]

        step = nperseg // 2
        n_segments = (n_times - nperseg) // step + 1

        from numpy.lib.stride_tricks import sliding_window_view

        windows = sliding_window_view(
            trial,
            window_shape=nperseg,
            axis=-1
        )[:, ::step, :]

        windows = windows * np.hanning(nperseg)

        Zxx = np.fft.rfft(
            windows,
            axis=-1
        )

        Pxx = np.mean(
            np.abs(Zxx) ** 2,
            axis=1
        )

        Pxy = np.einsum(
            'isf,jsf->ijf',
            Zxx,
            np.conj(Zxx)
        ) / n_segments

        Pxx_prod = (
            Pxx[:, np.newaxis, :]
            * Pxx[np.newaxis, :, :]
        )

        coh_matrix = (
            np.abs(Pxy) ** 2
            / (Pxx_prod + 1e-12)
        )

        f = np.fft.rfftfreq(
            nperseg,
            d=1 / sfreq
        )

        trial_features = []

        for ch1_idx, ch2_idx in pair_indices:
            pair_coh = coh_matrix[
                ch1_idx,
                ch2_idx,
                :
            ]

            for low, high in bands:
                mask = (
                    (f >= low)
                    & (f < high)
                )

                if np.any(mask):
                    trial_features.append(
                        np.mean(pair_coh[mask])
                    )
                else:
                    trial_features.append(0.0)

        coherence_features[t] = trial_features

        if (t + 1) % 50 == 0 or t == 0:
            print(
                f"  Coherence进度: "
                f"{t + 1}/{n_trials}"
            )

    print(
        f"Coherence特征完成: "
        f"shape = {coherence_features.shape}"
    )

    return coherence_features


# ================= 5. PLV =================
def calc_plv(X_filt, sfreq=500):
    bands = [(4, 8), (8, 13), (13, 30)]

    ch_names = [
        "FP1", "FP2", "Fz", "F3", "F4", "F7", "F8", "FCz",
        "FC3", "FC4", "FT7", "FT8", "Cz", "C3", "C4", "T3",
        "T4", "CPz", "CP3", "CP4", "TP7", "TP8", "Pz", "P3",
        "P4", "T5", "T6", "Oz", "O1", "O2", "HEOL", "HEOR"
    ]

    selected_pairs = [
        ("FP1", "FP2"),
        ("F3", "F4"),
        ("F7", "F8"),
        ("Fz", "Cz"),
        ("Fz", "Pz"),
        ("F3", "P3"),
        ("F4", "P4"),
        ("C3", "C4"),
        ("P3", "P4"),
        ("O1", "O2")
    ]

    ch_to_idx = {
        name: i for i, name in enumerate(ch_names)
    }

    pair_indices = [
        (ch_to_idx[ch1], ch_to_idx[ch2])
        for ch1, ch2 in selected_pairs
    ]

    n_trials, n_channels, n_times = X_filt.shape
    n_features = len(selected_pairs) * len(bands)

    plv_features = np.zeros(
        (n_trials, n_features),
        dtype=np.float64
    )

    print(
        f"正在计算 PLV: "
        f"{n_trials} 个 Trial × "
        f"{len(selected_pairs)} 个通道对 × "
        f"{len(bands)} 个频段"
    )

    print("选用的10对电极：")
    for i, (ch1, ch2) in enumerate(selected_pairs, 1):
        print(f"  {i}. {ch1}-{ch2}")

    for band_idx, (low, high) in enumerate(bands):
        print(
            f"  正在计算频段 "
            f"{low}-{high} Hz ..."
        )

        sos = butter(
            4,
            [low, high],
            btype='bandpass',
            fs=sfreq,
            output='sos'
        )

        X_band = sosfiltfilt(
            sos,
            X_filt,
            axis=2
        )

        analytic_signal = hilbert(
            X_band,
            axis=2
        )

        phase = np.angle(
            analytic_signal
        )

        for pair_idx, (ch1_idx, ch2_idx) in enumerate(pair_indices):
            phase_diff = (
                phase[:, ch1_idx, :]
                - phase[:, ch2_idx, :]
            )

            plv = np.abs(
                np.mean(
                    np.exp(1j * phase_diff),
                    axis=1
                )
            )

            feature_idx = (
                pair_idx * len(bands)
                + band_idx
            )

            plv_features[:, feature_idx] = plv

    print(
        f"PLV特征完成: "
        f"shape = {plv_features.shape}"
    )

    return plv_features


# ================= 6. 微状态特征 =================
def calc_microstate(X_filt, sfreq=500, n_states=4):
    """
    微状态特征：
    1. 平均持续时间：n_states维
    2. 出现频率：n_states维
    3. 时间覆盖率：n_states维
    4. 状态转换概率：n_states*n_states维

    n_states=4时，输出28维。
    """

    n_trials, n_channels, n_times = X_filt.shape

    # 只使用前30个头皮 EEG 通道，排除 HEOL、HEOR
    eeg_data = X_filt[:, :30, :]

    print("\n========== 开始微状态特征提取 ==========")
    print(f"Trial数量: {n_trials}")
    print(f"EEG通道数量: {eeg_data.shape[1]}")
    print(f"微状态数量: {n_states}")

    # ================= 1. 提取 GFP 峰值地形图 =================
    all_maps = []

    for t in range(n_trials):
        trial = eeg_data[t]

        # 计算 GFP
        gfp = np.std(trial, axis=0)

        # 寻找 GFP 峰值
        peaks, _ = find_peaks(
            gfp,
            distance=max(1, int(sfreq * 0.01))
        )

        if len(peaks) == 0:
            peaks = np.array([np.argmax(gfp)])

        # 提取峰值对应的地形图
        maps = trial[:, peaks]

        # 去除每张地形图的空间平均值
        maps = maps - np.mean(
            maps,
            axis=0,
            keepdims=True
        )

        # 极性不变处理
        max_idx = np.argmax(
            np.abs(maps),
            axis=0
        )

        signs = np.sign(
            maps[
                max_idx,
                np.arange(maps.shape[1])
            ]
        )

        signs[signs == 0] = 1

        maps = maps * signs[np.newaxis, :]

        all_maps.append(maps.T)

    all_maps = np.vstack(all_maps)

    print(
        f"GFP峰值地形图数量: {all_maps.shape[0]}"
    )

    # ================= 2. K-means聚类 =================
    map_norms = np.linalg.norm(
        all_maps,
        axis=1,
        keepdims=True
    )

    normalized_maps = all_maps / (
        map_norms + 1e-12
    )

    kmeans = KMeans(
        n_clusters=n_states,
        random_state=42,
        n_init=10
    )

    kmeans.fit(normalized_maps)

    # 获取微状态模板
    templates = kmeans.cluster_centers_

    templates = templates / (
        np.linalg.norm(
            templates,
            axis=1,
            keepdims=True
        ) + 1e-12
    )

    print("K-means微状态聚类完成！")
    print("开始计算每个Trial的微状态特征...")

    # ================= 3. 每个时间点分配微状态 =================
    microstate_features = []

    for t in range(n_trials):
        trial = eeg_data[t]

        # 去除每个时间点的空间平均值
        maps = trial - np.mean(
            trial,
            axis=0,
            keepdims=True
        )

        # 地形图归一化
        norms = np.linalg.norm(
            maps,
            axis=0,
            keepdims=True
        )

        normalized_trial = maps / (
            norms + 1e-12
        )

        # 计算地形图与模板的空间相关性
        correlations = templates @ normalized_trial

        # 使用绝对相关性，考虑极性不变性
        labels = np.argmax(
            np.abs(correlations),
            axis=0
        )

        # ================= 4. 持续时间、频率、覆盖率 =================
        durations = np.zeros(n_states)
        occurrences = np.zeros(n_states)
        coverages = np.zeros(n_states)

        trial_duration = n_times / sfreq

        # 找到微状态发生变化的位置
        change_points = (
            np.where(np.diff(labels) != 0)[0] + 1
        )

        boundaries = np.concatenate([
            [0],
            change_points,
            [n_times]
        ])

        state_durations = [
            [] for _ in range(n_states)
        ]

        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]

            state = labels[start]

            segment_duration = (
                end - start
            ) / sfreq

            state_durations[state].append(
                segment_duration
            )

        for state in range(n_states):
            # 平均持续时间
            if len(state_durations[state]) > 0:
                durations[state] = np.mean(
                    state_durations[state]
                )

            # 每秒出现次数
            occurrences[state] = (
                len(state_durations[state])
                / trial_duration
            )

            # 时间覆盖率
            coverages[state] = (
                np.sum(labels == state)
                / n_times
            )

        # ================= 5. 状态转换概率 =================
        transition_counts = np.zeros(
            (n_states, n_states)
        )

        for i in range(len(labels) - 1):
            current_state = labels[i]
            next_state = labels[i + 1]

            if current_state != next_state:
                transition_counts[
                    current_state,
                    next_state
                ] += 1

        # 按每种当前状态的总转换次数归一化
        row_sums = transition_counts.sum(
            axis=1,
            keepdims=True
        )

        transition_probs = np.divide(
            transition_counts,
            row_sums,
            out=np.zeros_like(transition_counts),
            where=row_sums != 0
        )

        # ================= 6. 合并28维特征 =================
        trial_features = np.concatenate([
            durations,
            occurrences,
            coverages,
            transition_probs.flatten()
        ])

        microstate_features.append(
            trial_features
        )

        if (t + 1) % 500 == 0 or t == 0:
            print(
                f"  微状态进度: "
                f"{t + 1}/{n_trials}"
            )

    microstate_features = np.asarray(
        microstate_features
    )

    print(
        f"微状态特征完成: "
        f"shape = {microstate_features.shape}"
    )

    print("========== 微状态特征提取结束 ==========\n")

    return microstate_features
# ================= 7. 空间协方差 =================
def calc_covariance(X_filt):
    """
    空间协方差特征：
    1. 每个Trial计算一个通道协方差矩阵
    2. 提取上三角部分（包含对角线）
    3. 展开为一维特征向量

    61个通道 -> 1891维特征
    """

    n_trials, n_channels, n_times = X_filt.shape

    # 获取上三角索引，包含对角线
    upper_idx = np.triu_indices(n_channels)

    n_features = len(upper_idx[0])

    covariance_features = np.zeros(
        (n_trials, n_features),
        dtype=np.float64
    )

    print("\n========== 开始空间协方差特征提取 ==========")
    print(f"Trial数量: {n_trials}")
    print(f"通道数量: {n_channels}")
    print(f"每个Trial特征维度: {n_features}")

    for t in range(n_trials):

        # 计算当前Trial的通道协方差矩阵
        cov_matrix = np.cov(
            X_filt[t],
            rowvar=True,
            ddof=1
        )

        # 提取上三角部分并展开
        covariance_features[t] = cov_matrix[upper_idx]

        if (t + 1) % 500 == 0 or t == 0:
            print(
                f"  Covariance进度: "
                f"{t + 1}/{n_trials}"
            )

    print(
        f"空间协方差特征完成: "
        f"shape = {covariance_features.shape}"
    )

    print("========== 空间协方差特征提取结束 ==========\n")

    return covariance_features
# ================= 7. 小波特征 =================
def calc_wavelet(X_filt, wavelet='db4', level=6):
    """
    小波能量特征：
    1. 使用 db4 小波
    2. DWT 分解 6 层
    3. 提取 D4、D5、D6、A6
    4. 计算各层系数的能量
    """

    n_trials, n_channels, n_times = X_filt.shape

    wavelet_features = np.zeros(
        (n_trials, n_channels * 4),
        dtype=np.float64
    )

    print("\n========== 开始小波特征提取 ==========")
    print(f"小波: {wavelet}")
    print(f"分解层数: {level}")
    print("提取: D4、D5、D6、A6")
    print(f"Trial数量: {n_trials}")
    print(f"通道数量: {n_channels}")

    for t in range(n_trials):
        trial_features = []

        for c in range(n_channels):
            signal = X_filt[t, c, :]

            coeffs = pywt.wavedec(
                signal,
                wavelet,
                level=level
            )

            # coeffs:
            # [A6, D6, D5, D4, D3, D2, D1]
            A6, D6, D5, D4 = coeffs[:4]

            # 小波能量 = 系数平方和
            energies = [
                np.sum(D4 ** 2),
                np.sum(D5 ** 2),
                np.sum(D6 ** 2),
                np.sum(A6 ** 2)
            ]

            trial_features.extend(energies)

        wavelet_features[t] = trial_features

        if (t + 1) % 500 == 0 or t == 0:
            print(
                f"  Wavelet进度: "
                f"{t + 1}/{n_trials}"
            )

    print(
        f"小波特征完成: "
        f"shape = {wavelet_features.shape}"
    )

    print("========== 小波特征提取结束 ==========\n")

    return wavelet_features


# ================= 7. 统一调度中心 =================
def extract_selected_features(
    X_filt,
    feature_list,
    sfreq=500
):
    all_feats = []

    # 1. 频域特征
    needs_freq = any(
        f in feature_list
        for f in ['PSD', 'DE', 'Rel']
    )

    if needs_freq:
        X_psd, X_de, X_rel = calc_psd_de_rel(
            X_filt,
            sfreq
        )

        if 'PSD' in feature_list:
            all_feats.append(X_psd)

        if 'DE' in feature_list:
            all_feats.append(X_de)

        if 'Rel' in feature_list:
            all_feats.append(X_rel)

    # 2. 非线性特征
    if 'Hjorth' in feature_list:
        all_feats.append(
            calc_hjorth(X_filt)
        )

    if 'PE' in feature_list:
        all_feats.append(
            calc_perm_entropy(X_filt)
        )

    # 3. Coherence
    if 'Coherence' in feature_list:
        all_feats.append(
            calc_coherence(
                X_filt,
                sfreq
            )
        )

    # 4. PLV
    if 'PLV' in feature_list:
        all_feats.append(
            calc_plv(
                X_filt,
                sfreq
            )
        )

    # 5. 微状态
    if 'Microstate' in feature_list:
        all_feats.append(
            calc_microstate(
                X_filt,
                sfreq,
                n_states=4
            )
        )
# 6. 小波
    if 'Wavelet' in feature_list:
         all_feats.append(
            calc_wavelet(
            X_filt,
            wavelet='db4',
            level=6
         )
    )

# 7. 空间协方差
    if 'Covariance' in feature_list:
        all_feats.append(
            calc_covariance(X_filt)
        )
    if not all_feats:
        raise ValueError(
            "特征列表为空，请至少选择一种特征！"
        )

    return np.hstack(all_feats)