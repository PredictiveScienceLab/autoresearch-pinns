# Symmetry Run Structure Change

The symmetry run did not win by scaling the network again. It kept the same DeepONet skeleton and mostly changed the representation so the known finite-domain symmetry is built in.

Baseline control (`26ecc84`, `val_rel_l2 = 1.6677972e-02`):
- `DeepONet` with `resmlp` branch / `mlp` trunk
- hidden widths `640` / `640`, latent `448`
- generic `fourier` trunk coordinates, raw initial-condition branch input
- no exact symmetry projection and no boundary envelope

Best kept symmetry model (`d4f60d6`, `val_rel_l2 = 9.7437697e-03`):
- same `DeepONet` branch/trunk families and the same hidden widths / latent size
- parity-aware `fourier_parity` trunk coordinates
- `reflection_sign_split` branch input so the model sees the mirror-sign partner explicitly
- exact `reflection_sign_projection` at prediction time
- `quadratic_dirichlet` envelope so the output respects the bounded Dirichlet walls

What changed structurally:
- coordinate encoding: `fourier` -> `fourier_parity`
- branch input mode: `raw` -> `reflection_sign_split`
- symmetry mode: `none` -> `reflection_sign_projection`
- boundary mode: `none` -> `quadratic_dirichlet`
- parameter count: `5.296M` -> `5.502M` (`+206,080` params, `3.9%`)

What stayed fixed:
- branch/trunk families stayed `resmlp` / `mlp`
- hidden widths stayed `640` / `640`
- latent size stayed `448`
- Fourier feature count and scale stayed `96` and `3.0`

Short read: the improvement came from imposing the correct bounded-domain mirror-sign structure, not from making the network much larger.
