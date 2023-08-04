import click

from raving_fader.cli import train, train_prior_attr


@click.group()
def main():
    pass


main.command()(train)
main.command()(train_prior_attr)


if __name__ == "__main__":
    main()
