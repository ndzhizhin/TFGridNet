import torch
import torchaudio.compliance.kaldi as kaldi
import onnxruntime as ort
import torchaudio

def extract_embedding(filename, model_path):
    """
    Извлекает эмбеддинги из аудиофайла.

    Args:
        filename: Аудиофайл.
        model_path: Путь к ONNX-модели.
    """

    session = ort.InferenceSession(model_path)

    sampling_rate = 16000

    emb, sr = torchaudio.load(filename, normalize=False)

    emb = emb.to(torch.float)

    if sr != sampling_rate:
        emb = torchaudio.transforms.Resample(
            orig_freq=sr, new_freq=sampling_rate)(emb)

    feats = compute_fbank(emb, sample_rate=sampling_rate, cmn=True)
    feats = feats.unsqueeze(0)
    emb = session.run(None, {"feats": feats.numpy()})[0]
    print(emb)
    return emb

def compute_fbank(wavform,
                  sample_rate=16000,
                  num_mel_bins=80,
                  frame_length=25,
                  frame_shift=10,
                  cmn=True):

    feat = kaldi.fbank(wavform,
                       num_mel_bins=num_mel_bins,
                       frame_length=frame_length,
                       frame_shift=frame_shift,
                       sample_frequency=sample_rate)
    if cmn:
        feat = feat - torch.mean(feat, 0)
    return feat

def preproccess_emb(emb_audio_path):
    emb_input = extract_embedding(emb_audio_path, r"onnx_resnet/resnet34.onnx")
    return emb_input