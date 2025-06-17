# Rubin Telescope Chatbot Panel

## Quick Start

To quickly launch a GitHub Codespace with Rubin data preloaded, 
<a title="Create New Rubin Codespace" href="https://github.com/codespaces/new?hide_repo_select=true&ref=main&repo=866827957&skip_quickstart=true&machine=standardLinux32gb&devcontainer_path=.devcontainer%2FScipy2024%2Fdevcontainer.json&geo=UsWest" target="_blank">click here</a>.

## About

This repository hosts the Chatbot Panel app designed to answer questions related to the Rubin Telescope.

## Instructions

1. **Launch GitHub Codespace**  
   Start the environment by <a title="Create New Rubin Codespace" href="https://github.com/codespaces/new?hide_repo_select=true&ref=main&repo=866827957&skip_quickstart=true&machine=standardLinux32gb&devcontainer_path=.devcontainer%2FScipy2024%2Fdevcontainer.json&geo=UsWest" target="_blank">clicking here</a>.  
   Alternatively, you can manually launch a codespace by following these steps:
   
   <div style="display: flex; gap: 10px;">
       <img src="images/open_with_codespaces.png" alt="Open with Codespaces" width="500" />
       <img src="images/select_dev_container.png" alt="Select Dev Container" width="500" />
   </div>

2. **Run the Panel Application**  
   Start the chatbot panel app by running the following command:
   ```bash
   pixi run python rubin-chat/rubin-panel-app.py

## GitHub Codespaces

This repository is available to run within
[Github Codespaces](https://github.com/features/codespaces) - "a development
environment that's hosted in the cloud".

A codespace is a development environment that's hosted in the cloud. GitHub
currently gives every user
[120 vCPU hours per month for free](https://docs.github.com/en/billing/managing-billing-for-github-codespaces/about-billing-for-github-codespaces#monthly-included-storage-and-core-hours-for-personal-accounts),
beyond that you must pay. **So be sure to explicitly stop or shut down your
codespace when you are done by going to this page
([https://github.com/codespaces](https://github.com/codespaces)).**


## Contributing

If you would like to contribute to this repository, please feel free to open up
issues or pull requests, regarding current or future materials.


## Acknowledgements

We would like to thank
[Professor Andy Connolly](https://escience.washington.edu/member/andy-connolly/)
for providing the datasets and the domain expertise for building this tool

We also thank the [Allen Institute for AI (AI2)](https://allenai.org/) for
providing the Open Language Model (OLMo)

This repository is based on a [tutorial](https://github.com/uw-ssec/tutorials/tree/main) from  engineers at the
[University of Washington's Scientific Software Engineering Center (SSEC)](https://escience.washington.edu/software-engineering/ssec/)
supported by Schmidt Sciences, LLC. as part of the
[Virtual Institutes for Scientific Software (VISS)](https://www.schmidtsciences.org/viss/).

<div>
  <a href="https://escience.washington.edu/software-engineering/ssec/">
    <img src="https://uw-ssec-tutorials.readthedocs.io/en/latest/_static/logo.png"
    alt="SSEC_logo" width="100">
  </a>
  <a href="https://www.schmidtsciences.org/viss/">
    <img src="https://www.schmidtsciences.org/wp-content/themes/schmidt-sciences/images/logo.png" alt="Schmidt_logo" width="150" style="background-color: #1a1e2d;">
  </a>
  <a href="https://escience.washington.edu/">
    <img src="https://escience.washington.edu/wp-content/uploads/2022/07/escience-logo-768x193.png" alt="eScience_logo" width="150" style="background-color: #ffffff;">
  </a>
  <a href="https://washington.edu/">
    <img src="https://upload.wikimedia.org/wikipedia/commons/1/17/Washington_Huskies_logo.svg" alt="uw_logo" height="100">
  </a>
</div>


## Running the Rubin-RAG App Locally

To run the Rubin-RAG panel app on your local machine (macOS), follow these steps:

### 1. Clone the repository (if not already done)
```bash
git clone -b clean https://github.com/baisakhi771/Rubin-RAG.git
cd Rubin-RAG
```

### 2. Set up a virtual environment (recommended)
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies from `requirements.txt`
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. (Optional) Install and use Pixi for environment management

Install Pixi (if not already installed):
```bash
curl -sSL https://pixi.sh/install.sh | bash
```

Then use it to install and manage dependencies:
```bash
pixi install
```

You can now run the app using Pixi:
```bash
pixi run python rubin-chat/rubin-panel-app.py
```

---



### 5. Ensure writable Qdrant directory

Update `rubin-panel-app.py`:
```python
from pathlib import Path

qdrant_path = Path("resources/rubin_qdrant")
```

Create the directory if it doesn’t exist:

```bash
mkdir -p resources/rubin_qdrant
```

---

### 6. Run the app

```bash
pixi run python rubin-chat/rubin-panel-app.py
```

Then visit [http://localhost:8000](http://localhost:8000) (Use the port number as shown on the terminal) in your browser to use the Rubin RAG interface.
