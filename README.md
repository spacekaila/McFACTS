<p align="center">
    <img width="500", src="https://www.google.com">
    <!-- put our logo here instead of google -->
</p>

<h2 align="center">
    The <b>M</b>onte <b>c</b>arlo <b>F</b>or <b>A</b>GN <b>C</b>hannel <b>T</b>esting and <b>S</b>imulations
    <br>
    <!-- <a href="https://github.com/TeamLEGWORK/LEGWORK-paper">
        <img src="https://img.shields.io/badge/release paper-repo-blue.svg?style=flat&logo=GitHub" alt="Read the article"/>
    </a>
    <a href="https://codecov.io/gh/TeamLEGWORK/LEGWORK">
        <img src="https://codecov.io/gh/TeamLEGWORK/LEGWORK/branch/main/graph/badge.svg?token=FUG4RFYCWX"/>
    </a>
    <a href='https://legwork.readthedocs.io/en/latest/?badge=latest'>
        <img src='https://readthedocs.org/projects/legwork/badge/?version=latest' alt='Documentation Status' />
    </a>
    <a href="https://ascl.net/2111.007">
        <img src="https://img.shields.io/badge/ascl-2111.007-blue.svg?colorB=262255" alt="ascl:2111.007" />
    </a>
    <a href="mailto:tomjwagg@gmail.com?cc=kbreivik@flatironinstitute.org">
        <img src="https://img.shields.io/badge/contact-authors-blueviolet.svg?style=flat" alt="Email the authors"/>
    </a> -->
</h2>

<p align="center">
    A python package that does the AGN channel for you!
</p>

### Installation
.. warning::

            We don't guarantee that there won't be mistakes or bugs in the development version, use at your own risk!

        The latest development version is available directly from our `GitHub Repo
        <https://github.com/bmckernan/test_mcfacts>`_. To start, clone the repository onto your machine:

        .. code-block:: bash
        
            git clone https://github.com/bmckernan/test_mcfacts
            cd test_mcfacts

        Next, we recommend that you create a Conda environment for working with McFACTS.
        You can do this by running

        .. code-block:: bash

            conda create --name mcfacts-dev "python>=3.10.4" pip "numpy>=1.23.1" "scipy>=1.11.2" "matplotlib>=3.5.2" -c conda-forge -c defaults
<!-- I don't know what the -c stuff does or if it is needed/actually wrong I copied it from LEGWORK -->

        And then activate the environment by running

        .. code-block:: bash

            conda activate mcfacts-dev

        At this point, all that's left to do is install McFACTS!

        .. code-block:: bash

            pip install .
<!-- Put simply? `pip install legwork`! But we recommend creating a conda environment first to ensure everything goes smoothly! Check out the installation instructions [here](https://legwork.readthedocs.io/en/latest/install.html) to learn exactly how to install LEGWORK -->

McFACTS has a couple of dependencies: `numpy`, `scipy`, `matplotlib`.
  <!-- (see [requirements.txt](requirements.txt) for the exact version requirements). -->

### Documentation
All documentation related to McFACTS can be found in the code for now.
<!-- [at this link](https://legwork.readthedocs.io/en/latest/) -->

<!-- ### Other quick links
- [Quickstart](https://legwork.readthedocs.io/en/latest/notebooks/Quickstart.html) - New to LEGWORK? Try out our quickstart tutorial!
- [Tutorials](https://legwork.readthedocs.io/en/latest/tutorials.html) - Learn more about what you can do with LEGWORK with our tutorials!
- [Citing LEGWORK](https://legwork.readthedocs.io/en/latest/cite.html) - If you're using LEGWORK for a scientific publication please follow the link for citation intstructions
- [Demos](https://legwork.readthedocs.io/en/latest/demos.html) - Want to see what LEGWORK is capable of? Check out our demos!
- [API reference](https://legwork.readthedocs.io/en/latest/modules.html) - Wondering how you should use a particular function? Go take a look at our full API reference!
- [Feature requests](https://github.com/TeamLEGWORK/LEGWORK/issues/new) - Do you have an idea for adding something to LEGWORK? Create an issue [here](https://github.com/TeamLEGWORK/LEGWORK/issues/new) and let us know! Or, even better, make the change yourself and create a [pull request](https://github.com/TeamLEGWORK/LEGWORK/pulls)!
- [Bug reporting](https://github.com/TeamLEGWORK/LEGWORK/issues/new) - If you see a bug we would love to know about it! Please create an issue [here](https://github.com/TeamLEGWORK/LEGWORK/issues/new)!
- [Release paper](https://arxiv.org/abs/2111.08717) - The LEGWORK release paper is now on the ArXiv and you can also view it directly [in GitHub](https://github.com/TeamLEGWORK/LEGWORK-paper) if you prefer! -->