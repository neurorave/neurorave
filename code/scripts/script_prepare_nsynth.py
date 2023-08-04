import os
import click
import json
import shutil
from glob import glob
from tqdm import tqdm


@click.argument("input_dir")
@click.argument("output_dir")
@click.option("--instru", "-i", default="string")
def nsynth(input_dir, output_dir, instru):
    os.makedirs(output_dir, exist_ok=True)

    for f in tqdm(glob(os.path.join(input_dir, "*.wav"))):
        file_name = os.path.basename(f)
        if file_name.startswith(instru):
            shutil.copyfile(
                src=f,
                dst=os.path.join(output_dir, file_name)
            )


# DEPRECATED
@click.argument("nsynth_dir")
@click.argument("output_dir")
@click.option("--instru", "-i", default="string")
def write_nsynth_json(nsynth_dir, output_dir, instru):
    with open(os.path.join(nsynth_dir, "examples.json"), "r") as f:
        data = json.load(f)

    data_strings = {k: v for k, v in data.items() if k.startswith(instru)}
    print(f"nb samples : {len(data_strings.keys())}")

    with open(os.path.join(output_dir, f"nsynth_{instru}.json"), "w", encoding="utf-8") as f:
        json.dump(data_strings, f, indent=4)


@click.group()
def main():
    pass


main.command()(nsynth)
main.command()(write_nsynth_json)


if __name__ == "__main__":
    main()
