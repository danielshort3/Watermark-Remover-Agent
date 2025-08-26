Place order of worship PDFs in this directory.

Files stored here can be referenced by passing their path to the
``run_order_graph`` function in ``watermark_remover.agent.order_graph`` or
providing the ``pdf_path`` key to the order graph state.  PDF files
should list songs in a recognisable format such as "Title by Artist in Key"
or "Title - Artist - Key".  The parser will ignore lines that do not
match these patterns.