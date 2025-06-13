#!/usr/bin/env python3
"""
Script to add papers from BibTeX file to README.md
Parses plasma_physics_ml_bibliography.bib and adds entries to README.md
"""

import re
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def parse_bibtex_entry(entry: str) -> Dict[str, str]:
    """Parse a single BibTeX entry and extract key information."""
    # Extract entry type and key
    match = re.match(r'@(\w+)\{([^,]+),', entry)
    if not match:
        return {}
    
    entry_type = match.group(1)
    key = match.group(2)
    
    # Extract fields
    fields = {}
    
    # Title
    title_match = re.search(r'title\s*=\s*\{([^}]+)\}', entry)
    if title_match:
        fields['title'] = title_match.group(1)
    
    # Authors - keep all authors without truncation
    author_match = re.search(r'author\s*=\s*\{([^}]+)\}', entry)
    if author_match:
        authors = author_match.group(1)
        # Split by 'and' and clean up
        author_list = [author.strip() for author in authors.split(' and ')]
        # Keep all authors without truncation
        fields['authors'] = ", ".join(author_list)
    
    # Year
    year_match = re.search(r'year\s*=\s*\{([^}]+)\}', entry)
    if year_match:
        fields['year'] = year_match.group(1)
    
    # ArXiv ID
    eprint_match = re.search(r'eprint\s*=\s*\{([^}]+)\}', entry)
    if eprint_match:
        fields['arxiv_id'] = eprint_match.group(1)
        fields['arxiv_url'] = f"https://arxiv.org/abs/{eprint_match.group(1)}"
    
    # DOI
    doi_match = re.search(r'doi\s*=\s*\{([^}]+)\}', entry)
    if doi_match:
        fields['doi'] = doi_match.group(1)
    
    # URL
    url_match = re.search(r'url\s*=\s*\{([^}]+)\}', entry)
    if url_match:
        fields['url'] = url_match.group(1)
    
    # Abstract - handle multi-line and nested braces
    abstract_match = re.search(r'abstract\s*=\s*\{', entry)
    if abstract_match:
        # Find the matching closing brace
        start_pos = abstract_match.end() - 1  # Position of opening brace
        brace_count = 1
        end_pos = start_pos + 1
        
        while end_pos < len(entry) and brace_count > 0:
            if entry[end_pos] == '{':
                brace_count += 1
            elif entry[end_pos] == '}':
                brace_count -= 1
            end_pos += 1
        
        if brace_count == 0:
            abstract_content = entry[start_pos + 1:end_pos - 1]
            fields['abstract'] = abstract_content.strip()
    
    fields['entry_type'] = entry_type
    fields['key'] = key
    
    return fields


def parse_bibtex_file(bibtex_path: Path) -> List[Dict[str, str]]:
    """Parse the entire BibTeX file and return list of entries."""
    with open(bibtex_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split into individual entries
    entries = []
    current_entry = ""
    in_entry = False
    brace_count = 0
    
    for line in content.split('\n'):
        line = line.strip()
        if not line or line.startswith('%'):
            continue
            
        if line.startswith('@'):
            if current_entry and in_entry:
                entries.append(parse_bibtex_entry(current_entry))
            current_entry = line
            in_entry = True
            brace_count = line.count('{') - line.count('}')
        elif in_entry:
            current_entry += '\n' + line
            brace_count += line.count('{') - line.count('}')
            
            if brace_count == 0:
                entries.append(parse_bibtex_entry(current_entry))
                current_entry = ""
                in_entry = False
    
    # Handle last entry
    if current_entry and in_entry:
        entries.append(parse_bibtex_entry(current_entry))
    
    return [entry for entry in entries if entry]  # Filter out empty entries


def format_readme_entry(entry: Dict[str, str]) -> str:
    """Format a BibTeX entry for README.md."""
    if not entry.get('title') or not entry.get('authors'):
        return ""
    
    title = entry['title']
    authors = entry['authors']
    year = entry.get('year', '')
    
    # Determine publication type and create appropriate text
    pub_type = "arXiv"
    if entry.get('arxiv_url'):
        link = f"[arXiv]({entry['arxiv_url']})"
    elif entry.get('url'):
        link = f"[link]({entry['url']})"
    else:
        link = "Link not available"
        pub_type = "journal"
    
    # Add abstract if available
    abstract_text = ""
    if entry.get('abstract'):
        # Truncate abstract if too long for README readability
        abstract = entry['abstract']
        if len(abstract) > 300:
            abstract = abstract[:297] + "..."
        abstract_text = f" - {abstract}"
    
    # Format: **year** - **title** - *authors* - arxiv / journal / conference - link - abstract or summary
    formatted = f"- **{year}** - **{title}** - *{authors}* - {pub_type} - {link}{abstract_text} <!-- imported-from-bib -->"
    
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
    new_papers_section = "## Research Papers\n\n*Papers are organized chronologically*\n\n**Format:** \"**year** - **title** - *authors* - arxiv / journal / conference - link - abstract or summary\n\n"
    
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