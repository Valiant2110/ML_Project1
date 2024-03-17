from setuptools import find_packages,setup


HYPEN_E_DOT='-e .'
def get_req(file_path:str)->list[str]:
    '''
    this function give the str from the txt in list
    '''
    req = []
    with open(file_path) as li:
        req = li.readlines()
        req = [re.replace("\n","") for re in req]
        
        if HYPEN_E_DOT in req:
            req.remove(HYPEN_E_DOT)
            
    return req
    

setup(
name = 'Ml-Project',
version='0.0.1',
author='Vipul',
author_email='vipul.v.gotiwale@gmail.com',
packages=find_packages(),
## install_requires =['pandas','numpy','seaborn','matplotlib','random','tensorflow'] 
# we can use this but we might require more of the file in future to make it future proof we will define the function
install_requires = get_req('requirements.txt')
)

