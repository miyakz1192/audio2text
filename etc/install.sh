set -x

cd /etc/systemd/system 
sudo rm audio2text.service
sudo ln -s ~/audio2text/etc/audio2text.service audio2text.service

sudo systemctl enable audio2text.service

