'''Cloud ML Engine package configuration.'''
from setuptools import setup, find_packages

REQUIRED_PACKAGES = [
    'keras>=2.2.4',
    'Pillow'
]

setup(name='sunny_inception_resnet_v2',
      version='2.1',
      packages=find_packages(),
      include_package_data=True,
      description='Train model on Cloud ML Engine',
      author='SHEN SHUTAO',
      author_email='sstwood@gmail.com',
      license='MIT',
      install_requires=REQUIRED_PACKAGES,
      zip_safe=False)
