import streamlit as st
import parselmouth
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import io
import tempfile
import os
import soundfile as sf

# 1. 페이지 설정
st.set_page_config(page_title="AI 음성 분석 시스템", layout="wide")
st.title("🎙️ AI 정밀 음성/영상 분석 시스템")
st.write("컴퓨터공학과 프로젝트: WAV 및 MP4 파일의 8가지 핵심 음향 지표 분석")

# 2. 파일 업로드 섹션
uploaded_file = st.file_uploader("음성(WAV) 또는 영상(MP4) 파일을 업로드하세요", type=["wav", "mp4", "m4a"])

if uploaded_file is not None:
    # 확장자 확인 및 임시 파일 생성
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    
    # 원본 파일 임시 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    # Parselmouth용 변환 파일 경로
    audio_only_path = tmp_file_path + "_converted.wav"

    try:
        with st.spinner('오디오 추출 및 정밀 분석 중...'):
            # [A] 오디오 추출 및 로드
            y, sr = librosa.load(tmp_file_path, sr=22050) # 안정적인 분석을 위해 sr 고정 권장
            
            # Parselmouth(Praat)와의 호환성을 위해 WAV로 저장
            sf.write(audio_only_path, y, sr)
            
            # 분석 객체 생성
            snd = parselmouth.Sound(audio_only_path)
            pitch = snd.to_pitch()
            formant = snd.to_formant_burg()
            point_process = parselmouth.praat.call([snd, pitch], "To PointProcess (cc)")
            harmonicity = snd.to_harmonicity_cc()

            # [B] 8가지 핵심 지표 추출 (안전한 수치 추출 로직)
            def get_praat_val(call_obj):
                val = parselmouth.praat.call(call_obj[0], call_obj[1], *call_obj[2:])
                return 0 if np.isnan(val) else val

            # 1. Pitch
            m_pitch = get_praat_val([pitch, "Get mean", 0, 0, "Hertz"])
            
            # 2. Formants (F1, F2, F3)
            f1 = get_praat_val([formant, "Get mean", 1, 0, 0, "Hertz"])
            f2 = get_praat_val([formant, "Get mean", 2, 0, 0, "Hertz"])
            f3 = get_praat_val([formant, "Get mean", 3, 0, 0, "Hertz"])
            
            # 3. Stability (Jitter, Shimmer)
            try:
                jitter = parselmouth.praat.call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
                shimmer = parselmouth.praat.call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            except:
                jitter, shimmer = 0, 0
            
            # 4. Harmonics (HNR)
            hnr = get_praat_val([harmonicity, "Get mean", 0, 0])
            
            # 5. Energy Ratio (L/H)
            S = np.abs(librosa.stft(y))
            freqs = librosa.fft_frequencies(sr=sr)
            low_band = np.sum(S[freqs <= 1000])
            high_band = np.sum(S[freqs > 1000])
            lh_ratio = low_band / high_band if high_band > 0 else 0

            # [C] 결과 화면 구성
            col1, col2 = st.columns([1, 2])

            with col1:
                st.subheader("📊 분석 결과 수치")
                st.metric("평균 Pitch (F0)", f"{m_pitch:.2f} Hz")
                st.write("**포먼트 (Formants)**")
                st.write(f"- F1 (입 크기/개구도): {f1:.2f} Hz")
                st.write(f"- F2 (혀 위치/전후): {f2:.2f} Hz")
                st.write(f"- F3 (음색 선명도): {f3:.2f} Hz")
                st.write("---")
                st.write("**음성 안정성**")
                st.write(f"- Jitter (주파수 떨림): {jitter*100:.3f}%")
                st.write(f"- Shimmer (진폭 떨림): {shimmer*100:.3f}%")
                st.write(f"- HNR (소음 대비 배음비): {hnr:.2f} dB")
                st.write(f"- L/H 에너지 비율: {lh_ratio:.4f}")

            with col2:
                st.subheader("📈 시각화 리포트")
                fig, ax = plt.subplots(2, 1, figsize=(10, 8))
                
                # 스펙트로그램
                D = librosa.amplitude_to_db(S, ref=np.max)
                librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=ax[0])
                ax[0].set_title("Spectrogram (Frequency Analysis)")

                # 포먼트 변화 그래프 (F1, F2, F3)
                times = pitch.xs()
                f1_vals = [formant.get_value_at_time(1, t) for t in times]
                f2_vals = [formant.get_value_at_time(2, t) for t in times]
                f3_vals = [formant.get_value_at_time(3, t) for t in times]
                
                ax[1].plot(times, f1_vals, label='F1', color='red', alpha=0.6)
                ax[1].plot(times, f2_vals, label='F2', color='green', alpha=0.6)
                ax[1].plot(times, f3_vals, label='F3', color='orange', alpha=0.6)
                ax[1].set_title("Formant Tracking Flow")
                ax[1].set_ylabel("Frequency (Hz)")
                ax[1].legend()

                plt.tight_layout()
                st.pyplot(fig)

        st.success("모든 분석이 성공적으로 완료되었습니다!")

    except Exception as e:
        st.error(f"분석 중 오류가 발생했습니다: {e}")
        st.info("파일 형식이 올바른지, 혹은 너무 짧은 파일은 아닌지 확인해 주세요.")

    finally:
        # 임시 파일들 삭제 (서버 용량 관리)
        for p in [tmp_file_path, audio_only_path]:
            if p and os.path.exists(p):
                os.remove(p)