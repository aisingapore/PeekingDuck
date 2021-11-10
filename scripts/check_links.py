"""Checks for faulty links in documentation"""

import urllib.request
from pathlib import Path
from typing import Iterator, List, Tuple, Union

import markdown
from bs4 import BeautifulSoup
from texttable import Texttable


# Currently not in use because Sphinx generated html files
# do not show in the Peekingduck repo on github
#
# def get_html():
#     # search path is hard coded with ref of this script locaiton
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


def check_for_faulty_links(
    file_paths: List[Path],
) -> List[Tuple[Path, str, Union[int, Path]]]:
    """Returns faulty links from documentation files in the repository.

    Parse the provided .md and .rst files for faulty hyperlinks or faulty
    relative path links. For URLs, the current implementation only returns links
    which give HTTP 404.

    Args:
        file_paths (List[Path]): File paths of all .md and .rst files in the
            repository.

    Returns:
        (List[Tuple[Path, str, Union[int, Path]]]): A list of file paths in
            which faulty links are found, the corresponding faulty links, and
            the root folder/request error code.
    """
    faulty_links: List[Tuple[Path, str, Union[int, Path]]] = []
    for path in file_paths:
        print(f"===== Checking {path}")
        with open(path, "r", encoding="utf-8") as infile:
            content = infile.read()
            content_html = markdown.markdown(content)
            soup = BeautifulSoup(content_html, "html.parser")
            img_links = [
                tag["src"]
                for tag in soup.find_all(
                    lambda tag: tag.name == "img" and tag.get("src")
                )
            ]
            href_links = [
                tag["href"]
                for tag in soup.find_all(
                    lambda tag: tag.name == "a" and tag.get("href")
                )
            ]
            # "." filters out section links, split("#")[0] to filter out URI
            # fragments
            all_links = [
                link.split("#")[0]
                for link in filter(lambda link: "." in link, img_links + href_links)
            ]
        for link in all_links:
            if link.startswith("http"):
                try:
                    # Validated the URL to start with "http"
                    urllib.request.urlopen(link)  # nosec
                except urllib.error.HTTPError as error:
                    # In this implementation only 404 is flagged for broken links
                    # 404 = http page not found error
                    # if statement can be removed/adjusted to flag multiple error
                    # codes such as 404,403,408...
                    if error.code == 404:
                        # path is the current file being parsed
                        # link is the link found in the current parsed file
                        # e.code is the execption code
                        rel_path = path.relative_to(Path.cwd())
                        faulty_links.append((rel_path, link, error.code))
            else:
                if not (path.parent / link).exists():
                    # path is the current file being parsed
                    # link is the link found in the current parsed file
                    # root is the root folder of the filepath of current file
                    condition = ["/peekingduck", "pipeline", "nodes"]
                    if link.split(".")[:3] != condition:
                        rel_path = path.relative_to(Path.cwd())
                        faulty_links.append((rel_path, link, rel_path.parent))
        print(f"Checked {path}")
    return faulty_links


def print_output(faulty_links: List[Tuple[Path, str, Union[int, Path]]]) -> None:
    """Displays the list of file paths and faulty links in a table

    Args:
        faulty_links (List[Tuple[Path, str, Union[int, Path]]]): A list of file
            paths in which faulty links are found, the corresponding faulty
            links, and the root folder/request error code.
    """
    print("\nTable of broken links\n")
    table = Texttable()
    table.set_cols_width([25, 25, 20])
    table.header(("Filepath", "Broken_Link", "Root_Folder / Request Error Code"))
    table.add_rows(faulty_links, False)
    print(table.draw())


if __name__ == "__main__":
    MD_RST_PATHS = get_md_rst_paths()
    print("\nCHECKING FILES")
    print("-" * 50)

    FAULTY_LINKS = check_for_faulty_links(sorted(MD_RST_PATHS))
    print_output(FAULTY_LINKS)
