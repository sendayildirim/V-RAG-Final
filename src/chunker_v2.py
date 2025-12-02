
from llama_index.core import Document
from llama_index.core.node_parser import HierarchicalNodeParser
from llama_index.core.node_parser import get_leaf_nodes
import re
from typing import List, Dict, Tuple

class HierarchicalChunker:
    """
    Hierarchical text chunker class using LlamaIndex HierarchicalNodeParser with chapter-based splitting
    Parent-child relationship is automatically created
    """

    def __init__(self, parent_size=2048, child_size=512, chunk_overlap=20):
        """
        Args:
            parent_size: Parent chunk size (characters)
            child_size: Child chunk size (characters)
            chunk_overlap: Overlap between chunks (characters)
        """
        self.parent_size = parent_size
        self.child_size = child_size
        self.chunk_overlap = chunk_overlap
        self.node_parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=[parent_size, child_size],
            chunk_overlap=chunk_overlap
        )

    def clean_gutenberg_text(self, text: str) -> str:  # clean book text

        # Take text after START marker
        start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK THE CHILDREN OF THE NEW FOREST ***"
        if start_marker in text:
            text = text.split(start_marker, 1)[1]

        # Take text before END marker
        end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK THE CHILDREN OF THE NEW FOREST ***"
        if end_marker in text:
            text = text.split(end_marker, 1)[0]

        # Clean extra spaces and line breaks
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)

        return text.strip()


    def parse_chapters(self, text: str) -> List[Tuple[str, str]]:  # split book into chapters

        # The Children of the New Forest chapter pattern: CHAPTER ONE., CHAPTER TWO., etc.
        chapter_pattern = re.compile(r'\n{2,}\s*CHAPTER\s+(ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN|ELEVEN|TWELVE|THIRTEEN|FOURTEEN|FIFTEEN|SIXTEEN|SEVENTEEN|EIGHTEEN|NINETEEN|TWENTY|TWENTY-ONE|TWENTY-TWO|TWENTY-THREE|TWENTY-FOUR|TWENTY-FIVE|TWENTY-SIX|TWENTY-SEVEN|TWENTY-EIGHT|TWENTY-NINE|THIRTY)\.\s*\n{2,}', re.MULTILINE | re.IGNORECASE)

        matches = list(chapter_pattern.finditer(text))

        if not matches:
            return [("CHAPTER ONE", text)]

        chapter_data = []

        # Capture chapters
        for idx, match in enumerate(matches):
            chapter_title = match.group(1).upper()  # ONE, TWO, etc
            start = match.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            chapter_text = text[start:end].strip()


            if chapter_text:
                chapter_data.append((f"CHAPTER {chapter_title}", chapter_text))
            else:
                print(f"WARNING: CHAPTER {chapter_title} has empty text, skipped")

        if not chapter_data and text.strip():
            chapter_data = [("CHAPTER ONE", text)]

        return chapter_data

    def chunk_text(self, text: str) -> Tuple[List, Dict]:
        """
        Chunk text hierarchically (chapter-based)

        Returns:
            (all_nodes, node_mapping) tuple
            - all_nodes: List of all nodes (parent + child)
            - node_mapping: node_id -> node mapping (for parent retrieval)
        """
        clean_text = self.clean_gutenberg_text(text)
        chapters = self.parse_chapters(clean_text)

        if not chapters:
            print("WARNING: No chapters found, processing entire text as single chapter")
            chapters = [("I", clean_text)]

        all_nodes = []
        node_mapping = {}

        for chapter_num, (chapter_title, chapter_text) in enumerate(chapters, 1):

            # LlamaIndex document
            doc = Document(
                text=chapter_text,
                metadata={
                    "chapter": chapter_num,
                    "chapter_title": chapter_title,
                    "book": "The Children of the New Forest"
                }
            )

            # Create hierarchical nodes (parent + child)
            nodes = self.node_parser.get_nodes_from_documents([doc])

            for node in nodes:
                node.metadata["chapter"] = chapter_num
                node.metadata["chapter_title"] = chapter_title
                node.metadata["book"] = "The Children of the New Forest"

                node_mapping[node.node_id] = node
                all_nodes.append(node)

        leaf_nodes = get_leaf_nodes(all_nodes)
        parent_nodes = [n for n in all_nodes if n not in leaf_nodes]

        print(f"Total {len(chapters)} chapters")
        print(f"Total {len(parent_nodes)} parent nodes created")
        print(f"Total {len(leaf_nodes)} child (leaf) nodes created")
        print(f"Total {len(all_nodes)} nodes")
        print(f"Chunk overlap: {self.chunk_overlap} characters")

        return all_nodes, node_mapping

    def get_chunk_stats(self, nodes: List) -> Dict:  # node statistics

        leaf_nodes = get_leaf_nodes(nodes)
        parent_nodes = [n for n in nodes if n not in leaf_nodes]

        leaf_lengths = [len(node.text) for node in leaf_nodes]
        parent_lengths = [len(node.text) for node in parent_nodes]

        chapters = {}
        for node in leaf_nodes:
            chapter = node.metadata.get('chapter', 0)
            if chapter not in chapters:
                chapters[chapter] = 0
            chapters[chapter] += 1

        stats = {
            'total_nodes': len(nodes),
            'total_parents': len(parent_nodes),
            'total_children': len(leaf_nodes),
            'avg_parent_length': sum(parent_lengths) / len(parent_lengths) if parent_lengths else 0,
            'avg_child_length': sum(leaf_lengths) / len(leaf_lengths) if leaf_lengths else 0,
            'max_child_length': max(leaf_lengths) if leaf_lengths else 0,
            'min_child_length': min(leaf_lengths) if leaf_lengths else 0,
            'num_chapters': len(chapters),
            'children_per_chapter': {k: v for k, v in chapters.items()},
            'chunk_overlap': self.chunk_overlap
        }

        return stats

if __name__ == "__main__":
    chunker = HierarchicalChunker(parent_size=2048, child_size=512, chunk_overlap=20)

    with open('data/children_of_new_forest.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    nodes, node_mapping = chunker.chunk_text(text)
    stats = chunker.get_chunk_stats(nodes)

    print("Node Statistics:")
    print(f"  Total nodes: {stats['total_nodes']}")
    print(f"  Parent nodes: {stats['total_parents']}")
    print(f"  Child nodes: {stats['total_children']}")
    print(f"  Number of chapters: {stats['num_chapters']}")
    print(f"  Chunk overlap: {stats['chunk_overlap']} characters")
    print(f"  Average parent length: {stats['avg_parent_length']:.1f} characters")
    print(f"  Average child length: {stats['avg_child_length']:.1f} characters")
    print(f"  Max child length: {stats['max_child_length']} characters")
    print(f"  Min child length: {stats['min_child_length']} characters")

    leaf_nodes = get_leaf_nodes(nodes)
    print(f"sample child node:")
    print(f"  Text: {leaf_nodes[0].text[:100]}")
    print(f"  Parent ID: {leaf_nodes[0].parent_node.node_id if leaf_nodes[0].parent_node else 'None'}")
    print(f"  Metadata: {leaf_nodes[0].metadata}")
