"""MedMNIST data-loader protocol with strict train/calibration/test separation."""
import argparse

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["pathmnist","dermamnist","bloodmnist","organamnist"], default="pathmnist")
    parser.add_argument("--size", type=int, default=28); parser.add_argument("--root", default="data")
    args=parser.parse_args()
    import medmnist
    info=medmnist.INFO[args.dataset]; cls=getattr(medmnist, info["python_class"])
    train=cls(split="train", root=args.root, size=args.size, download=True)
    calibration=cls(split="val", root=args.root, size=args.size, download=True)
    test=cls(split="test", root=args.root, size=args.size, download=True)
    print({"dataset":args.dataset,"classes":len(info["label"]),"train":len(train),"calibration":len(calibration),"test":len(test)})
    print("Use train only for fitting, val only for calibration/OOD thresholds, and test only for final triage evaluation.")
if __name__ == "__main__": main()
