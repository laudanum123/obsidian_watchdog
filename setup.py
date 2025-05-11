from setuptools import setup, find_packages

setup(
    name="obsidian_watchdog",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "openai",
        "pydantic-ai[openai]",
        "watchdog",
        "duckdb",
        "tinydb",
        "PyYAML",
    ],
) 