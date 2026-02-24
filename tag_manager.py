"""
Tag Management System for Memory System

This module handles:
- Extracting tags from memory content
- Normalizing tag variations
- Building and managing tag registries
- Maintaining canonical tag forms based on usage frequency

Used by database maintenance routines to keep tag registries synchronized
across main and short-term (Docker) systems.
"""

import re
import json
from typing import Dict, List, Set, Tuple
from collections import defaultdict
from pathlib import Path


class TagManager:
    """Manages tag extraction, normalization, and registry building."""

    # Delimiters used to split tag words
    TAG_WORD_DELIMITERS = [' ', '_', '-']
    
    # Regex pattern to extract tags from memory content
    TAG_PATTERN = r'\[Tags:\s*([^\]]+)\]'

    def __init__(self):
        """Initialize tag manager."""
        self.registry: Dict[str, Dict] = {}

    def extract_tags_from_content(self, content: str) -> List[str]:
        """
        Extract tags from memory content using regex pattern.
        
        Args:
            content: Memory content that may contain [Tags: tag1, tag2] format
            
        Returns:
            List of tag strings found, or empty list if no tags present
        """
        match = re.search(self.TAG_PATTERN, content)
        if not match:
            return []
        
        tags_str = match.group(1).strip()
        # Split on commas and clean up whitespace
        tags = [tag.strip() for tag in tags_str.split(',')]
        return [tag for tag in tags if tag]  # Remove empty strings

    def normalize_tag(self, tag: str) -> str:
        """
        Normalize a tag to lowercase for comparison.
        
        Args:
            tag: Raw tag string
            
        Returns:
            Normalized tag (lowercase)
        """
        return tag.lower().strip()

    def get_word_components(self, tag: str) -> List[str]:
        """
        Split tag into word components using delimiters.
        
        Args:
            tag: Tag string to split
            
        Returns:
            List of individual words
        """
        normalized = self.normalize_tag(tag)
        
        # Split on all delimiters
        words = [normalized]
        for delimiter in self.TAG_WORD_DELIMITERS:
            new_words = []
            for word in words:
                new_words.extend(word.split(delimiter))
            words = new_words
        
        # Remove empty strings and return unique words
        return list(set([w for w in words if w]))

    def build_tag_registry(self, memories: List[Dict]) -> Dict[str, Dict]:
        """
        Build complete tag registry from list of memories.
        
        Registry structure:
        {
            "canonical_form": {
                "canonical": "canonical_form",
                "variations": ["variation1", "variation2"],
                "word_components": ["word1", "word2"],
                "usage_count": 5
            }
        }
        
        Args:
            memories: List of memory dicts, each with 'content' and optionally 'tags' field
            
        Returns:
            Dictionary representing the tag registry
        """
        # Track all tag variations and their usage counts
        tag_variations: Dict[str, Set[str]] = defaultdict(set)
        tag_counts: Dict[str, int] = defaultdict(int)
        
        # Process each memory
        for memory in memories:
            content = memory.get('content', '')
            
            # Extract tags from content
            tags = self.extract_tags_from_content(content)
            
            # Also check if there's a tags field (JSON format)
            if 'tags' in memory and memory['tags']:
                if isinstance(memory['tags'], str):
                    try:
                        tags_from_field = json.loads(memory['tags'])
                        if isinstance(tags_from_field, list):
                            tags.extend(tags_from_field)
                    except json.JSONDecodeError:
                        pass
                elif isinstance(memory['tags'], list):
                    tags.extend(memory['tags'])
            
            # Deduplicate tags from this memory
            tags = list(set(tags))
            
            # Process each tag
            for tag in tags:
                normalized = self.normalize_tag(tag)
                tag_variations[normalized].add(tag)  # Store original variation
                tag_counts[normalized] += 1
        
        # Build registry with canonical forms (most frequent variation)
        registry = {}
        for normalized_tag, variations in tag_variations.items():
            # Find canonical form: most frequently used variation
            canonical = self._find_canonical_variation(normalized_tag, variations)
            
            registry[canonical.lower()] = {
                "canonical": canonical.lower(),
                "variations": sorted(list(variations)),
                "word_components": self.get_word_components(canonical),
                "usage_count": tag_counts[normalized_tag]
            }
        
        self.registry = registry
        return registry

    def _find_canonical_variation(self, normalized: str, variations: Set[str]) -> str:
        """
        Find the canonical form of a tag from its variations.
        
        Strategy: Use the most common variation among the provided set.
        If all have equal weight, prefer the original format.
        
        Args:
            normalized: Normalized form of the tag
            variations: Set of tag variations
            
        Returns:
            Canonical form of the tag
        """
        if not variations:
            return normalized
        
        # Convert to list for processing
        var_list = list(variations)
        
        # Prefer exact matches to normalized form first
        if normalized in var_list:
            return normalized
        
        # Then prefer snake_case or title case
        for v in var_list:
            if '_' in v or v[0].isupper():
                return v
        
        # Default to first one (arbitrary but consistent)
        return var_list[0]

    def save_registry(self, registry: Dict[str, Dict], filepath: str) -> bool:
        """
        Save tag registry to JSON file.
        
        Args:
            registry: Tag registry dictionary
            filepath: Path to save JSON file to
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure directory exists
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'w') as f:
                json.dump(registry, f, indent=2, sort_keys=True)
            
            return True
        except Exception as e:
            print(f"Error saving registry to {filepath}: {e}")
            return False

    def load_registry(self, filepath: str) -> Dict[str, Dict]:
        """
        Load tag registry from JSON file.
        
        Args:
            filepath: Path to registry JSON file
            
        Returns:
            Registry dictionary, or empty dict if file doesn't exist
        """
        try:
            if Path(filepath).exists():
                with open(filepath, 'r') as f:
                    self.registry = json.load(f)
                return self.registry
        except Exception as e:
            print(f"Error loading registry from {filepath}: {e}")
        
        return {}

    def get_canonical_form(self, tag: str) -> str:
        """
        Get the canonical form of a tag from the registry.
        
        Args:
            tag: Tag string to look up
            
        Returns:
            Canonical form if found, otherwise normalized input tag
        """
        normalized = self.normalize_tag(tag)
        
        if normalized in self.registry:
            return self.registry[normalized]["canonical"]
        
        # Check if the tag matches any variation
        for canonical_data in self.registry.values():
            if any(self.normalize_tag(v) == normalized 
                   for v in canonical_data["variations"]):
                return canonical_data["canonical"]
        
        # Not in registry, return normalized form
        return normalized

    def find_tag_by_any_variation(self, search_tag: str) -> Tuple[bool, str]:
        """
        Find if a tag exists in registry under any of its variations.
        
        Args:
            search_tag: Tag to search for (any variation)
            
        Returns:
            Tuple of (found: bool, canonical_form: str)
        """
        normalized = self.normalize_tag(search_tag)
        
        # Direct lookup
        if normalized in self.registry:
            return True, self.registry[normalized]["canonical"]
        
        # Check all variations
        for canonical_data in self.registry.values():
            if any(self.normalize_tag(v) == normalized 
                   for v in canonical_data["variations"]):
                return True, canonical_data["canonical"]
        
        return False, normalized

    def get_registry_summary(self) -> Dict:
        """
        Get a summary of the tag registry.
        
        Returns:
            Dict with total_tags, total_variations, and most_used tags
        """
        if not self.registry:
            return {
                "total_tags": 0,
                "total_variations": 0,
                "most_used": []
            }
        
        total_variations = sum(
            len(data["variations"]) for data in self.registry.values()
        )
        
        # Get top 10 most used tags
        most_used = sorted(
            [(data["canonical"], data["usage_count"]) 
             for data in self.registry.values()],
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        return {
            "total_tags": len(self.registry),
            "total_variations": total_variations,
            "most_used": most_used
        }
