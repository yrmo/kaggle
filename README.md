# kaggle

This repo contains my public Kaggle competition code. The aim is to be able to run the code locally, and in Kaggle notebooks, with zero modifications.

## Soft link

In Kaggle notebooks, the files are found at the root of the system at `/kaggle`, but generally I clone repos to the home directory `~/`. I use a symbolic link to map folder accesses at `/kaggle` to `~/kaggle` on our local machine. This is nice because it requires no code modifications for the code to work both on Kaggle and locally.

### Ubuntu

```sh
sudo ln -s ~/kaggle /kaggle
```

### MacOS

```sh
sudo touch /etc/synthetic.conf
sudo chmod 0666 /etc/synthetic.conf
echo "kaggle    Users/$(whoami)/kaggle" | sudo tee -a /etc/synthetic.conf # check the tab worked
sudo chmod 0644 /etc/synthetic.conf
sudo chown root:wheel /etc/synthetic.conf
# sudo reboot # necessary
```

## project.py

The entry point to performing actions in this repo. It's named `project.py` in the spirit of `pyproject.toml`. It uses [Google's Fire](https://github.com/google/python-fire) to automatically create the CLI interface. It uses the Kaggle API ([Kaggle](https://www.kaggle.com/docs/api), [GitHub](https://github.com/Kaggle/kaggle-api)) to interact with Kaggle, and can set environment variables for us automatically, which we use to conditionally change code behaviour to optimize for the task at hand. Each folder, besides `input` and `working` (output) in this repo, is a Kaggle competition. Each competition has a uniform API for interacting with it which you can see by running `python project.py`.

## Environment variables

Some variables are defined, so that in certain cases, our notebook can optimize for what action is being performed. While this does require some code modification, it is nice because we can write it in such a way that it'll still run in a Kaggle notebook if we are careful. I try to use these sparingly.

### `SUBMIT`

We are submitting the `submission.csv` to Kaggle, make choices that optimize submission score. If it is not set, set default to `'1'`, so uploaded Kaggle notebooks perform well.