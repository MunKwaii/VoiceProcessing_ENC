import librosa
import numpy as np
import os
from pydub import AudioSegment
import scipy
from matplotlib import pyplot as plt
from pydub import AudioSegment
import soundfile as sf
import config_params


def createpath(path):
    if not os.path.exists(path):
        os.makedirs(path)


def scale_dB(list_audio_files, dB_audio_dir, purepath=False):
    list_dB_sound = []

    for audio_file in list_audio_files:
        sound = AudioSegment.from_file(audio_file, format="wav")
        loudness = sound.dBFS
        change_in_dBFS = config_params.TARGET_dBFS - loudness

        scaled_audio = sound.apply_gain(change_in_dBFS)
        list_dB_sound.append(dB_audio_dir+audio_file.split('/')[-1])

        if (not purepath):
            scaled_audio.export(
                dB_audio_dir+audio_file.split('/')[-1], format="wav")
        else:
            scaled_audio.export(
                dB_audio_dir + "/" + audio_file.split('/')[-1], format="wav")

    return list_dB_sound


# def audio_to_audio_frame_stack(sound_data, frame_length, hop_length_frame):

#     sequence_sample_length = sound_data.shape[0]

#     sound_data_list = []
#     # Thuật toán cửa sổ trượt (Sliding Window)
#     for start in range(0, sequence_sample_length - frame_length + 1, hop_length_frame):
#         # Cắt lấy một đoạn tín hiệu dài 1 giây (16000 mẫu)
#         frame = sound_data[start : start + frame_length]
#         sound_data_list.append(frame)
#     sound_data_array = np.vstack(sound_data_list)

#     return sound_data_array


def audio_files_to_numpy(audio_files):
    list_sound = []

    signal, sr = librosa.load(audio_files, sr=None)

    # Only resmaple if the sample rate of file is not as specified
    if sr != config_params.SAMPLE_RATE:
        signal = librosa.resample(
            signal, orig_sr=sr, target_sr=config_params.SAMPLE_RATE, res_type='polyphase')

    list_sound.extend(signal)
    array_sound = np.array(list_sound)
    return array_sound

# def split_into_one_second(sound_data, save_dir, snr, category):
#     # get_duration: get the length of the secs
#     total_duration = librosa.get_duration(
#         y=sound_data, sr=config_params.SAMPLE_RATE)
#     splitted_audio_list = []
#     save_corpped_list = []
#     count = 0

#     if(category):
#         for time in range(0, int(total_duration)*16000, config_params.SLICE_LENGTH):
#             count += 1
#             start_time = time
#             end_time = time + config_params.SLICE_LENGTH
#             # print(total_duration, start_time, end_time)
#             SplittedAudio = sound_data[start_time:end_time]
#             if(int(SplittedAudio.shape[0]) < config_params.SLICE_LENGTH):
#                 continue
#             splitted_audio_list.append(SplittedAudio)
#             # print(save_dir, snr, category)
#             sf.write(save_dir + str(snr) + '/' + str(category) + '_' +
#                      str(count)+'.wav', SplittedAudio, config_params.SAMPLE_RATE, 'PCM_24')
#     else:
#         for time in range(0, int(total_duration)*16000, config_params.SLICE_LENGTH):
#             count += 1
#             start_time = time
#             end_time = time + config_params.SLICE_LENGTH
#             SplittedAudio = sound_data[start_time:end_time]
#             if(int(SplittedAudio.shape[0]) < config_params.SLICE_LENGTH):
#                 continue
#             splitted_audio_list.append(SplittedAudio)
#             sf.write(save_dir + str(snr) + '/' + str(count) + '.wav',
#                      SplittedAudio, config_params.SAMPLE_RATE, 'PCM_24')
#             save_corpped_list.extend(SplittedAudio)

#         sf.write(save_dir + str(snr) + '/' + 'Cropped.wav',
#                  np.vstack(save_corpped_list), config_params.SAMPLE_RATE, 'PCM_24')

#     return splitted_audio_list

def split_into_one_second(sound_data, save_dir, snr, category):
    # Lấy tổng thời lượng tệp âm thanh
    total_duration = librosa.get_duration(y=sound_data, sr=config_params.SAMPLE_RATE)
    splitted_audio_list = []
    
    # Vòng lặp băm nhỏ tín hiệu ngay trên bộ nhớ RAM
    # 16000 là số mẫu trong 1 giây (Sample Rate)
    for time in range(0, int(total_duration) * 16000, config_params.SLICE_LENGTH):
        start_time = time
        end_time = time + config_params.SLICE_LENGTH
        
        # Cắt lấy 1 giây dữ liệu
        SplittedAudio = sound_data[start_time : end_time]
        
        # Kiểm tra điều kiện đủ độ dài (1 giây) để nạp vào AI
        if int(SplittedAudio.shape[0]) < config_params.SLICE_LENGTH:
            continue
            
        # Chỉ thêm vào danh sách để xử lý trực tiếp trên RAM
        splitted_audio_list.append(SplittedAudio)
        
        # --- ĐÃ VÔ HIỆU HÓA PHẦN GHI FILE (Chỉ dùng khi Train) ---
        # sf.write(save_dir + ..., SplittedAudio, ...)

    return splitted_audio_list


def audio_to_magnitude_db_and_phase(n_fft, hop_length_fft, audio, i, path_save_image):

    # stftaudio = librosa.stft(
    #     audio, n_fft=n_fft, hop_length=hop_length_fft, window=scipy.signal.hamming)

    stftaudio = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length_fft, window='hamming')
    stftaudio_magnitude, stftaudio_phase = librosa.magphase(stftaudio)

    stftaudio_magnitude = stftaudio_magnitude[:, 1:-2]
    stftaudio_phase = stftaudio_phase[:, 1:-2]

    stftaudio_magnitude_db = librosa.amplitude_to_db(
        stftaudio_magnitude, ref=np.max)

    # print(stftaudio_magnitude)
    # print(np.abs(stftaudio))

    plt.imsave(os.path.join(path_save_image, str(i) + ".png"),
               stftaudio_magnitude_db, cmap='jet')

    # print(stftaudio_magnitude_db.shape, stftaudio_phase.shape)
    # print(stftaudio_magnitude_db.dtype, stftaudio_phase.dtype)

    return stftaudio_magnitude_db, stftaudio_phase


def numpy_audio_to_matrix_spectrogram(numpy_audio, path_save_image):
    '''
    Args:
        numpy_audio: the numpy array of audio
        path_save_image: str, path to save the spectrogram

    Returns:
        mag_db: magnitude in dB
        mag_phase: phase of magnitude
    '''
    nb_audio = numpy_audio.shape[0]

    mag_tmp, _ = audio_to_magnitude_db_and_phase(
        config_params.N_FFT, config_params.HOP_LENGTH_FFT, numpy_audio[0], 0, path_save_image)

    mag_shape = mag_tmp.shape

    mag_db = np.zeros((nb_audio, mag_shape[0], mag_shape[1]))
    mag_phase = np.zeros(
        (nb_audio, mag_shape[0], mag_shape[1]), dtype=complex)

    for i in range(nb_audio):
        mag_db[i, :, :], mag_phase[i, :, :] = audio_to_magnitude_db_and_phase(
            config_params.N_FFT, config_params.HOP_LENGTH_FFT, numpy_audio[i], i, path_save_image)

    return mag_db, mag_phase


def scaled_in(matrix_spec):
    "global scaling apply to noisy voice spectrograms (scale between -1 and 1)"
    matrix_spec = (matrix_spec + 46)/50
    return matrix_spec


# def scaled_ou(matrix_spec):
#     "global scaling apply to noise models spectrograms (scale between -1 and 1)"
#     matrix_spec = (matrix_spec - 6)/82
#     return matrix_spec


# def audio_files_add_to_numpy(list_audio_files):
#     list_sound = []

#     for file in list_audio_files:
#         signal, sr = librosa.load(file, sr=None)

#         # Only resmaple if the sample rate of file is not as specified
#         if sr != config_params.SAMPLE_RATE:
#             signal = librosa.resample(
#                 signal, orig_sr=sr, target_sr=config_params.SAMPLE_RATE, res_type='polyphase')

#         list_sound.extend(signal)

#     array_sound = np.array(list_sound)

#     return array_sound


def inv_scaled_ou(matrix_spec):

    matrix_spec = matrix_spec * 82 + 6
    return matrix_spec


def magnitude_db_and_phase_to_audio(hop_length_fft, stftaudio_magnitude_db, stftaudio_phase):

    stftaudio_magnitude_rev = librosa.db_to_amplitude(
        stftaudio_magnitude_db, ref=1.0)

    audio_reverse_stft = stftaudio_magnitude_rev * stftaudio_phase
    # audio_reconstruct = librosa.core.istft(
    #     audio_reverse_stft, hop_length=hop_length_fft, window=scipy.signal.hamming, center=True)
    
    audio_reconstruct = librosa.core.istft(audio_reverse_stft, hop_length=hop_length_fft, window='hamming', center=True)

    return audio_reconstruct


def matrix_spectrogram_to_numpy_audio(mag_db, mag_phase, fix_length, path_save_image):

    list_audio = []

    nb_spec = mag_db.shape[0]

    for i in range(nb_spec):

        plt.imsave(os.path.join(path_save_image, str(i) + ".png"),
                   mag_db[i], cmap='jet')

        audio_reconstruct = magnitude_db_and_phase_to_audio(
            config_params.HOP_LENGTH_FFT, mag_db[i], mag_phase[i])

        # audio_reconstruct = librosa.util.fix_length(
        #     audio_reconstruct, fix_length)
        
        audio_reconstruct = librosa.util.fix_length(audio_reconstruct, size=fix_length)

        list_audio.extend(audio_reconstruct)

    return np.vstack(list_audio)
