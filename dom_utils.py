import re
from bs4 import BeautifulSoup

def get_simplified_dom(html_content: str) -> str:
    """
    Simplifies the HTML content to make it easier for the LLM to process.
    It removes scripts, styles, and most non-interactive elements,
    and adds a unique 'agent-id-X' to each interactive element.

    Args:
        html_content: The raw HTML content of the page.

    Returns:
        A simplified string representation of the DOM.
    """
    soup = BeautifulSoup(html_content, 'html.parser')

    # Remove non-relevant tags
    for tag in soup(["script", "style", "meta", "link", "header", "footer", "nav", "aside"]):
        tag.decompose()

    # Find all interactive elements
    interactive_tags = soup.find_all([
        "a", "button", "input", "textarea", "select", "option"
    ])
    
    # Find all elements with an explicit role (e.g., role="button")
    role_based_interactive = soup.find_all(attrs={"role": re.compile(r"button|link|menuitem|tab|option|checkbox|radio|slider|textbox", re.IGNORECASE)})
    
    all_interactive = list(set(interactive_tags + role_based_interactive))

    # Add a unique agent-id to each interactive element
    for i, tag in enumerate(all_interactive):
        tag['agent-id'] = str(i)

    # Get the body or root, strip whitespace, and keep a limited amount of text
    if soup.body:
        body_content = soup.body.get_text(separator=' ', strip=True)
    else:
        body_content = soup.get_text(separator=' ', strip=True)
        
    # Limit the amount of raw text to avoid overwhelming the model
    page_text = ' '.join(body_content.split())[:3000]

    # Re-build the "simplified" HTML with only the interactive elements
    simplified_dom_parts = [f"PAGE_TEXT_SNIPPET: \"{page_text}\"", "\nINTERACTIVE_ELEMENTS:"]
    
    for tag in all_interactive:
        # Extract meaningful attributes
        attrs = {
            'agent-id': tag.get('agent-id'),
            'tag': tag.name,
            'text': ' '.join(tag.get_text(strip=True).split())[:200], # Limit text length
            'href': tag.get('href') if tag.name == 'a' else None,
            'type': tag.get('type') if tag.name == 'input' else None,
            'placeholder': tag.get('placeholder') if tag.name == 'input' else None,
            'aria-label': tag.get('aria-label'),
            'role': tag.get('role'),
        }
        # Filter out None values
        clean_attrs = {k: v for k, v in attrs.items() if v}
        simplified_dom_parts.append(str(clean_attrs))

    return "\n".join(simplified_dom_parts)
