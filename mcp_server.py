import base64
from pathlib import Path

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv(override=True)

MODEL = "gemini-3.5-flash-lite"          # modèle de génération (voir note si erreur 404)
EMBEDDING_MODEL = "gemini-embedding-001"  # modèle d'embeddings

# ------------------------------------------------------------------
# 1) Base de connaissances : vrais PDF (manuels/) + figures (figures/)
# ------------------------------------------------------------------
embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

# 1.a) Passages : chargement et découpage des PDF
loader = PyPDFDirectoryLoader(path="./manuels/")
documents = loader.load()
if not documents:
    raise RuntimeError("Aucun PDF trouvé dans ./manuels/")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)
print(f"{len(documents)} pages chargées, {len(chunks)} chunks créés")

passages_store = FAISS.from_documents(documents=chunks, embedding=embeddings)

# 1.b) Figures : fichiers image dans figures/ + description (fichier, manuel, page, légende)
FIGURES = [
    ("fig_3_1.png", "manuel_px200.pdf", 3, "Figure 3.1 - Voyant orange clignotant : bourrage papier"),
    ("fig_3_2.png", "manuel_px200.pdf", 3, "Figure 3.2 - Voyant rouge clignotant : niveau d'encre critique ou cartouche non reconnue"),
    ("fig_3_3.png", "manuel_px200.pdf", 3, "Figure 3.3 - Voyants orange et rouge en alternance : erreur système"),
    ("fig_4_1.png", "manuel_px200.pdf", 4, "Figure 4.1 - Pictogramme de la feuille froissée : bourrage papier"),
    ("fig_5_1.png", "manuel_px200.pdf", 5, "Figure 5.1 - Pictogramme de la goutte d'encre barrée : cartouche vide ou non reconnue"),
]
for fichier, _, _, _ in FIGURES:
    if not (Path("figures") / fichier).exists():
        print(f"ATTENTION : figures/{fichier} introuvable")

figures_store = FAISS.from_texts(
    texts=[legende for _, _, _, legende in FIGURES],
    embedding=embeddings,
    metadatas=[{"fichier": f, "source": s, "page": p} for f, s, p, _ in FIGURES],
)


# ------------------------------------------------------------------
# 2) Outils internes de l'agent RAG
# ------------------------------------------------------------------
@tool
def search_passages(query: str) -> str:
    """Recherche des passages de texte dans les manuels d'appareils"""
    docs = passages_store.similarity_search(query, k=3)
    return "\n".join(
        f"[{Path(d.metadata['source']).name} p.{int(d.metadata.get('page', 0)) + 1}] {d.page_content}"
        for d in docs
    )


@tool
def search_figures(query: str) -> str:
    """Recherche des figures (pictogrammes, schémas) dans les manuels d'appareils"""
    docs = figures_store.similarity_search(query, k=2)
    return "\n".join(
        f"[{d.metadata['source']} p.{d.metadata['page']} figures/{d.metadata['fichier']}] {d.page_content}"
        for d in docs
    )


llm = ChatGoogleGenerativeAI(model=MODEL, temperature=0)

rag_agent = create_agent(
    model=llm,
    tools=[search_passages, search_figures],
    system_prompt="""Tu es un agent RAG pour le SAV d'appareils électroniques.
Utilise search_passages et search_figures pour retrouver les éléments pertinents.
Réponds uniquement à partir de ces éléments et cite tes sources (document, page, figure).
Si l'information n'existe pas : "Je ne sais pas - information absente des manuels".""",
)


def charger_image(image_ref: str) -> dict | None:
    """Transforme une référence (ex: img_px200_01) en bloc image base64."""
    chemin = Path("images") / f"{image_ref}.png"
    if not chemin.exists():
        return None
    b64 = base64.b64encode(chemin.read_bytes()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}


# ------------------------------------------------------------------
# 3) Serveur MCP : outil distant rag_multimodal
# ------------------------------------------------------------------
mcp = FastMCP(name="mcp-server-sav", host="0.0.0.0", port=24000)


@mcp.tool()
def rag_multimodal(question: str, image_ref: str = ""):
    """Répond à une question de dépannage à partir des manuels (texte + figures).
    image_ref : référence optionnelle d'une photo de l'écran de l'appareil (ex: img_px200_01)"""
    content = [{"type": "text", "text": question}]
    if image_ref:
        bloc = charger_image(image_ref)
        if bloc:
            content.append(bloc)
        else:
            content[0]["text"] += f"\n(Photo jointe : {image_ref})"
    resp = rag_agent.invoke({"messages": [HumanMessage(content=content)]})
    return resp["messages"][-1].text


if __name__ == "__main__":
    mcp.run(transport="streamable-http")