<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/pone-software/apollo">
    <img src="https://www.pacific-neutrino.org/typo3conf/ext/sitepackage/Resources/Public/Images/Logos/P-ONE_Rainbow-01-360p.jpg" alt="Logo" height="143" width="360">
  </a>

<h3 align="center">Apollo</h3>

  <p align="center">
    P-One's Neural Network based trigger algorithm
    <br />
    <a href="https://pone-software.github.io/apollo/"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/pone-software/apollo/issues">Report Bug</a>
    ·
    <a href="https://github.com/pone-software/apollo/issues">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->

## About The Project

[![Product Name Screen Shot][product-screenshot]](https://example.com)

Like the greek god Apollo, this repository is all about predicting things.
The [Pacific Ocean Neutrino Experimen (P-ONE)](https://www.pacific-neutrino.org/) is a
neutrino detector to be built in the Pacific Ocean. High event rates and limited
bandwidth require a preselection of "interesting", in our case extragalactic high-energy
neutrino, events. This has to be done in real time as the data stream does not stop.
This repository contains neural network models that classify whether the current
timeframe contains something interesting or not based on simulated events of different
detector structures.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With

* [![Python][Python]][Python-url]
* [![PyTorch][PyTorch]][PyTorch-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->

## Getting Started

This section tells you how to set up the package and be able to run with it.

### Prerequisites

This package is built and updated using [Poetry](https://python-poetry.org/). 
Please install it and make yourself familiar if you never heard of it.

### Installation

To install the virtual environment of the package call the following console command

```console
foo@bar:apollo/$ poetry install
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->

## Usage

To use it, either call the commands of the `Makefile`. To list them call `make help`.
If you want to have a granular check, you can run

```console
foo@bar:apollo/$ poetry shell
```

To open a virtual environment.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ROADMAP -->

## Roadmap

### V0.0.1

* Create DataLoader
* Create NoisedDataLoader
* Create Notebook showing Data Loaders
* Cleaning up Examples Folder

### v1.0.0

* Set up ML Workflow
* Create first simple model
* Create runnable ML Algorithm
* Make package installable and add prerequisites and installation in template

See the [open issues](https://github.com/pone-software/apollo/issues) for a full list of
proposed features (and known
issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTRIBUTING -->

## Contributing

Contributions are what make the open source community such an amazing place to learn,
inspire, and create. Any
contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a
pull request. You can also
simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

### How to branch

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE.txt -->

## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->

## Contact

Janik Prottung - [@janikprottung](https://linkedin.com/in/janikprottung) -
janik.prottung@tum.de

Project
Link: [https://github.com/pone-software/apollo](https://github.com/pone-software/apollo)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->

## Acknowledgments

* [Best-README-Template](https://github.com/othneildrew/Best-README-Template)
* [Choose an Open Source License](https://choosealicense.com/)
* [Img Shields](https://shields.io/)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[contributors-shield]: https://img.shields.io/github/contributors/pone-software/apollo.svg?style=for-the-badge

[contributors-url]: https://github.com/pone-software/apollo/graphs/contributors

[forks-shield]: https://img.shields.io/github/forks/pone-software/apollo.svg?style=for-the-badge

[forks-url]: https://github.com/pone-software/apollo/network/members

[stars-shield]: https://img.shields.io/github/stars/pone-software/apollo.svg?style=for-the-badge

[stars-url]: https://github.com/pone-software/apollo/stargazers

[issues-shield]: https://img.shields.io/github/issues/pone-software/apollo.svg?style=for-the-badge

[issues-url]: https://github.com/pone-software/apollo/issues

[license-shield]: https://img.shields.io/github/license/pone-software/apollo.svg?style=for-the-badge

[license-url]: https://github.com/pone-software/apollo/blob/main/LICENSE.txt

[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555

[linkedin-url]: https://linkedin.com/in/janikprottung

[product-screenshot]: https://via.placeholder.com/1920x1080.png?text=Beautiful+Picture+To+Be+Done

[Python]: https://img.shields.io/badge/python-2b5b84?style=for-the-badge&logo=python&logoColor=white

[Python-url]: https://www.python.org/

[PyTorch]: https://img.shields.io/badge/PyTorch-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white

[PyTorch-url]: https://pytorch.org/