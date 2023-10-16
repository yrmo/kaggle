# ubuntu (untested)

```
sudo ln -s ~/kaggle /kaggle
```

# macos

```
sudo touch /etc/synthetic.conf
sudo chmod 0666 /etc/synthetic.conf
echo "kaggle    Users/$(whoami)/kaggle" | sudo tee -a /etc/synthetic.conf # check the tab worked
sudo chmod 0644 /etc/synthetic.conf
sudo chown root:wheel /etc/synthetic.conf
# sudo reboot
```
