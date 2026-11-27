"""Export each available checkpoint, retaining the original weights."""
import argparse
import torch
import onnx
from fer_models import ROOT, network


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--int8",action="store_true",help="Calibrate QDQ weights and activations on development images.")
    args = parser.parse_args()
    for kind,shape in (("cnn",(1,1,48,48)),("mobilenet",(1,3,96,96))):
        suffix = "_smoke" if args.smoke else ""
        path = ROOT/f"models/fer2013_{kind}{suffix}_best.pth"
        if not path.exists():
            print(f"Pending {path.name}")
            continue
        model = network(kind).eval()
        model.load_state_dict(torch.load(path,map_location="cpu"))
        dest = path.with_suffix(".onnx")
        torch.onnx.export(model,torch.zeros(shape),str(dest),input_names=["image"],
                          output_names=["logits"],opset_version=17,dynamo=False)
        onnx.checker.check_model(str(dest))
        print(dest)
        if args.int8:
            import pandas as pd
            from onnxruntime.quantization import CalibrationDataReader, QuantType, quantize_static
            from fer_models import RESULTS, Faces
            manifest = pd.read_csv(RESULTS/"fer_manifest.csv")
            rows = manifest[manifest.split=="train"].sample(64,random_state=42)
            calibration = Faces(rows,kind)

            class Reader(CalibrationDataReader):
                def __init__(self):
                    self.items = iter(range(len(calibration)))

                def get_next(self):
                    index = next(self.items,None)
                    if index is None:
                        return None
                    return {"image":calibration[index][0].unsqueeze(0).numpy()}

            quantize_static(str(dest),str(dest.with_suffix(".int8.onnx")),Reader(),
                            activation_type=QuantType.QUInt8,weight_type=QuantType.QInt8)


if __name__=="__main__":
    main()
