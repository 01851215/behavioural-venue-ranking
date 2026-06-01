from setuptools import setup, find_packages

setup(
    name="bvr",
    version="0.2.0",
    description="Behavioral Venue Ranking — emergence-aware bipartite graph ranking",
    author="Chris Liu",
    packages=find_packages(exclude=["tests*", "papers*", "docs*"]),
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "torch",
        "streamlit",
        "altair",
        "folium",
        "streamlit-folium",
        "fastapi",
        "uvicorn",
        "duckdb",
    ],
)
