### Set up Docker in Linux
* `ssh root@{ip}`
* Based on https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-on-ubuntu-20-04#step-1-installing-docker
    * `apt update`
    * `sudo apt install apt-transport-https ca-certificates curl software-properties-common`
    * `curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -`
    * `sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu focal stable"`
    * `apt-cache policy docker-ce`
    * Install Docker `sudo apt install docker-ce`
    * Check whether it is running`sudo systemctl status docker`
    * Install docker compose `apt install docker-compose`
### Pull data from github and install package in docker 
* `git clone {URL}`
* `cd {filename}`
* `docker-compose up -d`
* Check ip and port `docker ps`