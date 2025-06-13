#!/usr/bin/env python3
"""
Script to add papers from BibTeX file to README.md
Parses plasma_physics_ml_bibliography.bib and adds entries to README.md

Requires: pip install bibtexparser --pre
or: uv pip install bibtexparser --pre
"""

import re
import argparse
from pathlib import Path
from typing import Dict, List
import bibtexparser


def convert_bibtex_entry(entry) -> Dict[str, str]:
    """Convert a bibtexparser Entry to our internal format."""
    fields = {}
    
    # Extract basic fields
    fields['entry_type'] = entry.entry_type
    fields['key'] = entry.key
    
    # Title
    if 'title' in entry.fields_dict:
        fields['title'] = entry.fields_dict['title'].value
    
    # Authors - keep all authors without truncation
    if 'author' in entry.fields_dict:
        authors = entry.fields_dict['author'].value
        # Split by 'and' and clean up
        author_list = [author.strip() for author in authors.split(' and ')]
        # Keep all authors without truncation
        fields['authors'] = ", ".join(author_list)
    
    # Year
    if 'year' in entry.fields_dict:
        fields['year'] = entry.fields_dict['year'].value
    
    # ArXiv ID
    if 'eprint' in entry.fields_dict:
        arxiv_id = entry.fields_dict['eprint'].value
        fields['arxiv_id'] = arxiv_id
        fields['arxiv_url'] = f"https://arxiv.org/abs/{arxiv_id}"
    
    # DOI
    if 'doi' in entry.fields_dict:
        fields['doi'] = entry.fields_dict['doi'].value
    
    # URL
    if 'url' in entry.fields_dict:
        fields['url'] = entry.fields_dict['url'].value
    
    # Abstract
    if 'abstract' in entry.fields_dict:
        fields['abstract'] = entry.fields_dict['abstract'].value.strip()
    
    return fields


def parse_bibtex_file(bibtex_path: Path) -> List[Dict[str, str]]:
    """Parse the entire BibTeX file and return list of entries."""
    # Parse the BibTeX file
    library = bibtexparser.parse_file(str(bibtex_path))
    
    # Convert each entry to our format
    entries = []
    for entry in library.entries:
        converted = convert_bibtex_entry(entry)
        if converted:
            entries.append(converted)
    
    return entries


def format_readme_entry(entry: Dict[str, str]) -> str:
    """Format a BibTeX entry for README.md."""
    if not entry.get('title') or not entry.get('authors'):
        return ""
    
    title = entry['title']
    authors = entry['authors']
    year = entry.get('year', '')
    
    # Determine publication type and create appropriate link
    links = []
    
    # Check for arXiv
    if entry.get('arxiv_url'):
        links.append(f"[arXiv]({entry['arxiv_url']})")
    
    # Check for DOI
    if entry.get('doi'):
        doi_url = f"https://doi.org/{entry['doi']}"
        links.append(f"[DOI]({doi_url})")
    
    # Check for URL (only if it's not already an arXiv URL)
    if entry.get('url') and not (entry.get('arxiv_url') and entry['url'].startswith('http://arxiv.org')):
        url = entry['url']
        
        # Determine link text based on URL using match-case
        match url:
            case _ if 'arxiv.org' in url:
                link_text = 'arXiv'
            case _ if 'doi.org' in url or 'dx.doi.org' in url:
                link_text = 'DOI'
            case _ if 'ieee.org' in url:
                link_text = 'IEEE'
            case _ if 'nature.com' in url:
                link_text = 'Nature'
            case _ if 'sciencedirect.com' in url or 'elsevier.com' in url:
                link_text = 'ScienceDirect'
            case _ if 'aps.org' in url:
                link_text = 'APS'
            case _ if 'iop.org' in url or 'iopscience.iop.org' in url:
                link_text = 'IOP'
            case _ if 'aip.org' in url:
                link_text = 'AIP'
            case _ if 'openreview.net' in url:
                link_text = 'OpenReview'
            case _ if 'theses' in url or 'thesis' in url:
                link_text = 'Thesis'
            case _:
                link_text = 'Link'
        
        # Avoid duplicate DOI links
        if link_text != 'DOI' or not entry.get('doi'):
            links.append(f"[{link_text}]({url})")
    
    # Join all links
    if links:
        link_str = " | ".join(links)
    else:
        link_str = "No link available"
    
    # Determine publication type based on entry type using match-case
    match entry.get('entry_type', 'article'):
        case 'article':
            pub_type = 'journal'
        case 'inproceedings':
            pub_type = 'conference'
        case 'phdthesis' | 'mastersthesis':
            pub_type = 'thesis'
        case 'techreport':
            pub_type = 'technical report'
        case 'book':
            pub_type = 'book'
        case 'misc':
            pub_type = 'preprint'
        case _:
            pub_type = 'publication'
    
    # Add abstract if available
    abstract_text = ""
    abstract_len = 500
    if entry.get('abstract'):
        # Truncate abstract if too long for README readability
        abstract = entry['abstract']
        if len(abstract) > abstract_len:
            abstract = abstract[:abstract_len - 3] + "..."
        abstract_text = f" - {abstract}"
    
    # Format: **title** - **year** - *authors* - arxiv / journal / conference - link - abstract or summary
    formatted = f"**{title}** - **{year}** - *{authors}* - {pub_type} - {link_str}{abstract_text} <!-- imported-from-bib -->\n"
    
    return formatted


def get_year_from_entry(entry: Dict[str, str]) -> int:
    """Extract year from entry, default to 0 if not found."""
    try:
        return int(entry.get('year', '0'))
    except ValueError:
        return 0


def group_entries_by_year(entries: List[Dict[str, str]]) -> Dict[int, List[Dict[str, str]]]:
    """Group entries by year."""
    year_groups = {}
    for entry in entries:
        year = get_year_from_entry(entry)
        if year not in year_groups:
            year_groups[year] = []
        year_groups[year].append(entry)
    
    return year_groups


def parse_existing_readme_entries(papers_section: str) -> Dict[int, List[str]]:
    """Parse existing README entries and group by year."""
    year_entries = {}
    
    # Split by year sections
    year_sections = re.split(r'### (\d{4})', papers_section)[1:]  # Skip first empty part
    
    for i in range(0, len(year_sections), 2):
        if i + 1 < len(year_sections):
            year = int(year_sections[i])
            section_content = year_sections[i + 1]
            
            # Extract entries (lines starting with -)
            lines = section_content.split('\n')
            entries = []
            
            for line in lines:
                line = line.strip()
                if line.startswith('- ') and '<!-- imported-from-bib -->' not in line:
                    entries.append(line)
            
            if entries:
                year_entries[year] = entries
    
    return year_entries


def combine_entries_by_year(existing_entries: Dict[int, List[str]], 
                           new_entries: Dict[int, List[Dict[str, str]]]) -> Dict[int, Dict[str, List]]:
    """Combine existing and new entries by year."""
    combined = {}
    
    # Get all years
    all_years = set(existing_entries.keys()) | set(new_entries.keys())
    
    for year in all_years:
        combined[year] = {
            'existing': existing_entries.get(year, []),
            'new': new_entries.get(year, [])
        }
    
    return combined


def parse_existing_readme_entries_simple(papers_section: str) -> List[Dict[str, str]]:
    """Parse existing README entries into a simple list."""
    entries = []
    
    # Extract all lines that start with '-' and don't have imported-from-bib flag
    lines = papers_section.split('\n')
    
    for line in lines:
        line = line.strip()
        if line.startswith('- ') and '<!-- imported-from-bib -->' not in line:
            # Try to extract year from the line for sorting
            year = 0
            # Look for year patterns in the line
            year_match = re.search(r'\b(20\d{2})\b', line)
            if year_match:
                year = int(year_match.group(1))
            
            # Try to extract title for sorting
            title_match = re.search(r'\*\*([^*]+)\*\*', line)
            title = title_match.group(1) if title_match else ''
            
            entry = {
                'formatted': line,
                'year': year,
                'title': title,
                'is_imported': False
            }
            entries.append(entry)
    
    return entries


def remove_imported_entries(content: str) -> str:
    """Remove all entries with imported-from-bib flag from content."""
    # Remove lines containing imported-from-bib flag
    lines = content.split('\n')
    filtered_lines = []
    
    for line in lines:
        if '<!-- imported-from-bib -->' not in line:
            filtered_lines.append(line)
    
    return '\n'.join(filtered_lines)


def update_readme(readme_path: Path, entries: List[Dict[str, str]], dry_run: bool = False) -> None:
    """Update README.md with new entries."""
    with open(readme_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove existing imported entries
    content = remove_imported_entries(content)
    
    # Find the Research Papers section
    papers_section_match = re.search(r'## Research Papers.*?(?=## |\Z)', content, re.DOTALL)
    if not papers_section_match:
        print("Could not find Research Papers section in README.md")
        return
    
    papers_section = papers_section_match.group(0)
    
    # Parse existing entries (without year grouping)
    existing_entries = parse_existing_readme_entries_simple(papers_section)
    
    # Combine all entries and sort by year (descending) and title
    all_entries = []
    
    # Add existing entries
    all_entries.extend(existing_entries)
    
    # Add new entries from BibTeX
    for entry in entries:
        formatted_entry = format_readme_entry(entry)
        if formatted_entry:
            # Create entry dict with year for sorting
            entry_with_year = {
                'formatted': formatted_entry,
                'year': get_year_from_entry(entry),
                'title': entry.get('title', ''),
                'is_imported': True
            }
            all_entries.append(entry_with_year)
    
    # Sort all entries by year (descending) then by title
    all_entries.sort(key=lambda x: (-x.get('year', 0), x.get('title', '')))
    
    # Build new papers section
    new_papers_section = "## Research Papers\n\n*Papers are organized chronologically*\n\n**Format:** \"**title** - **year** - *authors* - journal/conference/thesis - link - abstract or summary\"\n\n"
    
    for entry in all_entries:
        new_papers_section += entry['formatted'] + "\n"
    
    # Replace the papers section
    new_content = content.replace(papers_section, new_papers_section)
    
    if dry_run:
        print("DRY RUN - Would update README.md with:")
        print("=" * 50)
        print(new_papers_section)
        print("=" * 50)
        
        # Count removed entries
        removed_count = content.count('<!-- imported-from-bib -->')
        print(f"\nWould remove {removed_count} existing imported entries")
        print(f"Would add {len(entries)} new entries from BibTeX")
    else:
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Updated README.md: removed old imported entries and added {len(entries)} new entries")


def main():
    parser = argparse.ArgumentParser(description='Add papers from BibTeX to README.md')
    parser.add_argument('--bibtex', type=Path, default='plasma_physics_ml_bibliography.bib',
                        help='Path to BibTeX file (default: plasma_physics_ml_bibliography.bib)')
    parser.add_argument('--readme', type=Path, default='README.md',
                        help='Path to README file (default: README.md)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be changed without actually modifying files')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not args.bibtex.exists():
        print(f"BibTeX file not found: {args.bibtex}")
        return
    
    if not args.readme.exists():
        print(f"README file not found: {args.readme}")
        return
    
    # Parse BibTeX file
    print(f"Parsing BibTeX file: {args.bibtex}")
    entries = parse_bibtex_file(args.bibtex)
    print(f"Found {len(entries)} entries")
    
    if not entries:
        print("No valid entries found in BibTeX file")
        return
    
    # Update README
    print(f"Updating README: {args.readme}")
    update_readme(args.readme, entries, dry_run=args.dry_run)


if __name__ == '__main__':
    main()