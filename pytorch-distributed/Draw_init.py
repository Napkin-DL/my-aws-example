      col = i % molsPerRow
      highlights = None
      if highlightAtomLists and highlightAtomLists[i]:
        highlights = highlightAtomLists[i]
      if mol is not None:
        img = _moltoimg(mol, subImgSize, highlights, legends[i], **kwargs)
        res.paste(img, (col * subImgSize[0], row * subImgSize[1]))
  else:
    fullSize = (molsPerRow * subImgSize[0], nRows * subImgSize[1])
    d2d = rdMolDraw2D.MolDraw2DCairo(fullSize[0], fullSize[1], subImgSize[0], subImgSize[1])
    d2d.DrawMolecules(
      list(mols), legends=legends, highlightAtoms=highlightAtomLists,
      highlightBonds=highlightBondLists, **kwargs)
    d2d.FinishDrawing()
    res = _drawerToImage(d2d)

  return res


def _MolsToGridSVG(mols, molsPerRow=3, subImgSize=(200, 200), legends=None, highlightAtomLists=None,
                   highlightBondLists=None, **kwargs):
  """ returns an SVG of the grid
  """
  if legends is None:
    legends = [''] * len(mols)

  nRows = len(mols) // molsPerRow
  if len(mols) % molsPerRow:
    nRows += 1

  blocks = [''] * (nRows * molsPerRow)

  fullSize = (molsPerRow * subImgSize[0], nRows * subImgSize[1])

  d2d = rdMolDraw2D.MolDraw2DSVG(fullSize[0], fullSize[1], subImgSize[0], subImgSize[1])
  d2d.DrawMolecules(mols, legends=legends, highlightAtoms=highlightAtomLists,
                    highlightBonds=highlightBondLists, **kwargs)
  d2d.FinishDrawing()
  res = d2d.GetDrawingText()
                                                                                                                                435,5         51%
