cd ../admin
source venv/bin/activate
cd ansible-playbooks
. 00-init.sh
ansible-playbook ray-cpu.yaml
