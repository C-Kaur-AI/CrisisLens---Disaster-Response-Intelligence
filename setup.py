from setuptools import setup, find_packages

setup(
    name="crisislens",
    version="0.1.0",
    description="Multilingual Crisis & Disaster Response NLP Pipeline",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="CrisisLens Team",
    license="Apache-2.0",
    python_requires=">=3.9",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "sentence-transformers>=2.2.0",
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "pydantic>=2.5.0",
        "pydantic-settings>=2.1.0",
        "geopy>=2.4.0",
        "emoji>=2.8.0",
        "folium>=0.15.0",
        "pandas>=2.1.0",
        "scikit-learn>=1.3.0",
    ],
    extras_require={
        "dev": ["pytest", "black", "ruff", "httpx"],
        "dashboard": ["streamlit", "streamlit-folium", "plotly"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
