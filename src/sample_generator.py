import mlflow


class GenerateSample:
    def __init__(self, model, i2w, sos_idx, eos_idx, device, mode, temperature):
        self.model = model
        self.i2w = i2w
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.device = device
        self.mode = mode
        self.temperature = temperature

    def __call__(self, i):
        gen = self.model.inference(
            self.sos_idx,
            self.eos_idx,
            device=self.device,
            mode=self.mode,
            temperature=self.temperature,
        )
        gen = [self.i2w[str(i)] for i in gen]
        s = "".join(gen[1:-1])
        headers = f"X:{i}\nL:1/8\nQ:120\nM:4/4\nK:C\n"
        tune = headers + s
        file_name = f"generated_sample_{i}.abc"

        # Return the file name for convenience
        return (file_name, tune)
