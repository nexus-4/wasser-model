"""Fine-tune de deteccao de gado em vista aerea.

Uso:
    uv run python training/train.py --epochs 100 --imgsz 1280

Saida: runs/detect/<name>/weights/best.pt
Aponte DEFAULT_MODEL_PATH (processor.py) para esse arquivo e passe
class_ids=[CATTLE_CLASS_ID] em process_video().
"""
import argparse
import sys
from pathlib import Path

import yaml
from ultralytics import YOLO

REPO = Path(__file__).resolve().parent.parent


def resolve_dataset_yaml(template_path):
    """Gera um yaml com caminhos absolutos a partir do template.

    O Ultralytics resolve um 'path:' relativo contra o datasets_dir global
    dele (~/datasets), nao contra a pasta do yaml -- e esse valor e lido no
    import, entao settings.update() em runtime nao adianta. Reescrevemos os
    caminhos como absolutos para o dataset.yaml continuar portavel entre
    maquinas.
    """
    template_path = Path(template_path).resolve()
    spec = yaml.safe_load(template_path.read_text(encoding="utf-8"))

    base = (template_path.parent / spec.get("path", ".")).resolve()
    spec["path"] = str(base)
    for split in ("train", "val", "test"):
        if split in spec:
            spec[split] = str((base / spec[split]).resolve())

    resolved = template_path.parent / ".dataset.resolved.yaml"
    resolved.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    return resolved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="yolo26s.pt",
                        help="modelo base. 's' treina rapido; 'x' da mais precisao e exige GPU.")
    parser.add_argument("--data", type=Path, default=REPO / "training" / "dataset.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=1280,
                        help="gado de drone e objeto pequeno; 1280 detecta muito mais que 640.")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default=None,
                        help="cuda / mps / cpu / 0. Padrao: detecta automaticamente.")
    parser.add_argument("--name", default="wasser-cattle")
    parser.add_argument("--workers", type=int, default=2,
                        help="processos de carga de dados. Baixe para 0-2 se der MemoryError.")
    parser.add_argument("--scale", type=float, default=0.5,
                        help="augmentacao de escala (+/-). NAO suba muito: o gado do "
                             "aerial-cows tem ~9px, e scale=0.9 encolhe ate 0.1x, "
                             "reduzindo o animal a menos de 1 pixel.")
    args = parser.parse_args()

    if not args.data.exists():
        raise SystemExit(f"dataset.yaml nao encontrado: {args.data}")

    data_yaml = resolve_dataset_yaml(args.data)

    sys.path.insert(0, str(REPO))
    from processor import resolve_device

    device = resolve_device(args.device)
    print(f"Device: {device}")
    if device == "cpu":
        print("AVISO: treino em CPU e lento. Com GPU NVIDIA instale o"
              " extra CUDA: uv sync --extra cu130")

    model = YOLO(args.base_model)
    results = model.train(
        data=str(data_yaml),
        # Sem project explicito o Ultralytics usa o runs_dir global dele, que
        # pode apontar para outro projeto da maquina.
        project=str(REPO / "runs"),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        name=args.name,
        # Vista aerea nao tem "cima" canonico: rotacao e flip vertical sao
        # aumentos validos aqui, ao contrario de fotos de nivel do chao.
        flipud=0.5,
        fliplr=0.5,
        degrees=180.0,
        # Tentador subir a escala para cobrir a diferenca de altitude entre
        # o dataset publico (~9px) e a nossa filmagem (~44px), mas nao
        # funciona: scale alto encolhe o animal para menos de 1 pixel e
        # destroi a amostra. A diferenca se resolve no fine-tune da etapa 2,
        # com frames proprios.
        scale=args.scale,
        workers=args.workers,
    )
    pesos = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\nTreino concluido. Pesos: {pesos}")
    print("Para usar: aponte DEFAULT_MODEL_PATH em processor.py para esse arquivo")
    print("e passe class_ids=[CATTLE_CLASS_ID] em process_video().")


if __name__ == "__main__":
    main()
