"""Checks for faulty links in documentation."""

from pathlib import Path
from typing import Iterator, List, Optional, Tuple, Union

import markdown
import requests
from bs4 import BeautifulSoup
from texttable import Texttable

# Currently not in use because Sphinx generated html files
# do not show in the Peekingduck repo on github
#
# def get_html():
#     # search path is hard coded with ref of this script location
#     return [
#         path
#         for path in walk(Path.cwd() / "docs" / "build" / "html")
#         if path.suffix == ".html"
#     ]


def walk(top: Path) -> Iterator[Path]:
    """Walks a given directory tree using pathlib.Path.

    Returns:
        (Iterator): An iterator of the file paths in the directory tree
    """
    for path in top.iterdir():
        if path.is_dir():
            yield from walk(path)
            continue
        yield path.resolve()


def get_md_rst_paths() -> List[Path]:
    """Returns all .md and .rst files in the repository"""
    return [path for path in walk(Path.cwd()) if path.suffix in (".md", ".rst")]


def check_link(link: str, path: Path) -> Optional[Tuple[Path, str, str]]:
    """Checks if the link/ref is valid. For HTTP response status codes
    handling, we flag all 4xx (client errors) for broken links.

    Args:
        link (str): An external link or a reference to another document that is
            found in the currently parsed file.
        path (Path): Path to the .md/.rst file containing that being parsed.

    Returns:
        (Optional[Tuple[Path, str, str]]): None when `link` is valid. A tuple
        containing (path of file being parsed, link/ref, root directory/error
        code) when the link/ref is faulty.
    """
    rel_path = path.relative_to(Path.cwd())

    def faulty_link_info(root_or_error: Union[int, str, Path]) -> Tuple[Path, str, str]:
        return rel_path, link, str(root_or_error)

    if link.startswith("http"):
        try:
            response = requests.get(link, timeout=10)
            if 400 <= response.status_code < 500:
                return faulty_link_info(response.status_code)
        except requests.exceptions.Timeout:
            return faulty_link_info("timed out")
        except requests.RequestException:
            pass
    else:
        if not (path.parent / link).exists():
            condition = ["/peekingduck", "pipeline", "nodes"]
            if link.split(".")[:3] != condition:
                rel_path = path.relative_to(Path.cwd())
                return faulty_link_info(rel_path.parent)
    return None


def check_for_faulty_links(file_paths: List[Path]) -> List[Tuple[Path, str, str]]:
    """Returns faulty links from documentation files in the repository.

    Parse the provided .md and .rst files for faulty hyperlinks or faulty
    relative path links. For URLs, the current implementation only returns
    links which give HTTP 404 and links which times out.

    Args:
        file_paths (List[Path]): File paths of all .md and .rst files in the
            repository.

    Returns:
        (List[Tuple[Path, str, str]]): A list of file paths in which faulty
        links are found, the corresponding faulty links, and the root
        folder/error code.
    """
    faulty_links: List[Tuple[Path, str, str]] = []
    for path in file_paths:
        content_html = markdown.markdown(path.read_text("utf-8"))
        soup = BeautifulSoup(content_html, "html.parser")
        img_links = [
            tag["src"]
            for tag in soup.find_all(lambda tag: tag.name == "img" and tag.get("src"))
        ]
        href_links = [
            tag["href"]
            for tag in soup.find_all(lambda tag: tag.name == "a" and tag.get("href"))
        ]
        # "." filters out section links, split("#")[0] to filter out URI
        # fragments
        all_links = [
            link.split("#")[0]
            for link in filter(lambda link: "." in link, img_links + href_links)
        ]
        for link in all_links:
            output = check_link(link, path)
            if output is not None:
                faulty_links.append(output)
        print(f"Checked {path}")
    return faulty_links


def print_output(faulty_links: List[Tuple[Path, str, str]]) -> None:
    """Displays the list of file paths and faulty links in a table

    Args:
        faulty_links (List[Tuple[Path, str, str]]): A list of file
            paths in which faulty links are found, the corresponding faulty
            links, and the root folder/request error code.
    """
    print("\nTable of broken links\n")
    table = Texttable()
    table.set_cols_width([25, 25, 20])
    table.header(("Filepath", "Broken Link", "Root Folder /\nError Code"))
    table.add_rows(faulty_links, False)
    print(table.draw())


if __name__ == "__main__":
    MD_RST_PATHS = get_md_rst_paths()
    print("\nCHECKING FILES")
    print("-" * 50)

    FAULTY_LINKS = check_for_faulty_links(sorted(MD_RST_PATHS))
    print_output(FAULTY_LINKS)
