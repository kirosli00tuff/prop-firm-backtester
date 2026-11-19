# Dependencies and licences

One row per package in uv.lock (exact versions as locked). Licences were read from the installed distribution
metadata (`importlib.metadata`: License-Expression, License, Classifier) and confirmed against the LICENSE file in
the dist-info where the metadata was empty or terse. Acceptable: MIT, BSD, Apache-2.0, ISC, PSF, LGPL. MPL-2.0 and
anything else counts as FLAG.

## FLAG list

- certifi 2026.10.23 (MPL-2.0): transitive runtime dependency of streamlit through requests; installed
  unmodified as a CA bundle. Flagged for the maintainer's decision.
- arch 9.0.0 (NCSA, the University of Illinois/NCSA Open Source Licence): direct runtime dependency for SPA, Reality Check and StepM (src/pfb/analysis/family_tests.py), used unmodified. NCSA is a
  permissive MIT/BSD-style licence (keep the copyright notice in source and binary redistributions, no
  endorsement by the author's name, no warranty) but is not on the acceptable list above, so it is flagged
  for the maintainer's decision. PyPI page for arch 9.0.0 (fetched 2026-11-17): License expression: "NCSA".
- scipy 1.18.1 (BSD-3-Clause; the wheel bundles OpenBLAS and LAPACK under BSD-3-Clause, the GCC runtime
  library under GPL-3.0-or-later WITH GCC-exception-3.1 and libquadmath under LGPL-2.1-or-later, per the
  `Name:`/`License:` lines of its LICENSE.txt): transitive runtime dependency of arch and statsmodels. The
  GCC runtime exception permits use from non-GPL code; flagged only because GPL-with-exception is not on
  the acceptable list.

colorama is locked for Windows only (`sys_platform == 'win32'`, required by pytest and others) and is not installed
on this machine; its licence was read from the wheel at the lock hash, fetched to a scratch directory outside the repo.

| package | version | direct or transitive, scope | licence | where read | class |
|---|---|---|---|---|---|
| annotated-doc | 0.0.6 | transitive, runtime | MIT | License-Expression; LICENSE in dist-info | PERMISSIVE |
| annotated-types | 0.8.0 | transitive, runtime | MIT | License-Expression, Classifier; LICENSE in dist-info | PERMISSIVE |
| altair | 6.3.0 | transitive, runtime | BSD-3-Clause | Classifier BSD License (License field is the text); LICENSE in dist-info is 3-clause (names may not endorse) | PERMISSIVE |
| anyio | 4.16.0 | transitive, runtime | MIT | License-Expression; LICENSE in dist-info | PERMISSIVE |
| arch | 9.0.0 | direct, runtime | NCSA | License-Expression; licenses/LICENSE.md in dist-info; PyPI page License expression "NCSA" | FLAG |
| attrs | 26.1.0 | transitive, runtime | MIT | License-Expression; LICENSE in dist-info | PERMISSIVE |
| certifi | 2026.10.23 | transitive, runtime | MPL-2.0 | License field MPL-2.0 + Classifier Mozilla Public License 2.0; LICENSE in dist-info (via requests, from streamlit) | FLAG |
| cffi | 2.1.1 | transitive, runtime | MIT-0 | License-Expression; LICENSE in dist-info (MIT No Attribution, looser than MIT) | PERMISSIVE |
| charset-normalizer | 3.5.2 | transitive, runtime | MIT | License field MIT; LICENSE in dist-info | PERMISSIVE |
| click | 8.5.1 | transitive, runtime | BSD-3-Clause | License-Expression; LICENSE.txt in dist-info | PERMISSIVE |
| colorama | 0.4.6 | transitive, runtime | BSD-3-Clause | Classifier BSD License in METADATA of the wheel at the lock hash (scratch install); LICENSE.txt confirms 3-clause; Windows-only, not installed here | PERMISSIVE |
| cryptography | 50.0.2 | transitive, runtime | Apache-2.0 OR BSD-3-Clause | License-Expression; LICENSE, LICENSE.APACHE and LICENSE.BSD in dist-info | PERMISSIVE |
| duckdb | 2.0.0 | direct, runtime | MIT | Classifier MIT License (no License field); LICENSE text confirms MIT | PERMISSIVE |
| formulaic | 1.2.2 | transitive, runtime (arch via statsmodels) | MIT | License-Expression, Classifier; licenses/LICENSE in dist-info | PERMISSIVE |
| h11 | 0.16.0 | transitive, runtime | MIT | License field MIT + Classifier; LICENSE.txt in dist-info | PERMISSIVE |
| httpcore2 | 2.16.2 | transitive, runtime | BSD-3-Clause | License-Expression + Classifier BSD License; LICENSE.md in dist-info | PERMISSIVE |
| httpx2 | 2.16.2 | transitive, runtime | BSD-3-Clause | License-Expression + Classifier BSD License; LICENSE.md in dist-info | PERMISSIVE |
| idna | 3.20 | transitive, runtime | BSD-3-Clause | License-Expression; LICENSE.md in dist-info | PERMISSIVE |
| interface-meta | 2.0.1 | transitive, runtime (arch via statsmodels, formulaic) | MIT | License field MIT; licenses/LICENSE in dist-info is MIT | PERMISSIVE |
| iniconfig | 2.3.1 | transitive, dev | MIT | License-Expression; LICENSE in dist-info | PERMISSIVE |
| itsdangerous | 2.2.0 | transitive, runtime | BSD-3-Clause | Classifier BSD License (no License field); LICENSE.txt in dist-info is 3-clause | PERMISSIVE |
| jinja2 | 3.1.6 | transitive, runtime | BSD-3-Clause | Classifier BSD License (no License field); LICENSE.txt in dist-info is 3-clause | PERMISSIVE |
| jsonschema | 4.26.0 | direct, runtime | MIT | License-Expression; COPYING in dist-info | PERMISSIVE |
| jsonschema-specifications | 2025.9.1 | transitive, runtime | MIT | License-Expression; COPYING in dist-info | PERMISSIVE |
| llvmlite | 0.50.0 | transitive, runtime | BSD-2-Clause AND Apache-2.0 WITH LLVM-exception | License-Expression; LICENSE (BSD-2) and LICENSE.thirdparty (Apache-2.0 with LLVM exception) in dist-info | PERMISSIVE |
| markdown-it-py | 4.3.0 | transitive, runtime | MIT | Classifier MIT License; LICENSE text confirms MIT | PERMISSIVE |
| markupsafe | 3.0.4 | transitive, runtime | BSD-3-Clause | License-Expression; LICENSE.txt in dist-info | PERMISSIVE |
| mcp | 2.3.0 | direct, runtime | MIT | License field MIT + Classifier MIT License; LICENSE in dist-info | PERMISSIVE |
| mcp-types | 2.3.0 | transitive, runtime | MIT | License field MIT + Classifier MIT License; LICENSE in dist-info | PERMISSIVE |
| mdurl | 0.1.2 | transitive, runtime | MIT | Classifier MIT License; LICENSE text confirms MIT | PERMISSIVE |
| narwhals | 2.28.0 | transitive, runtime | MIT | License-Expression; LICENSE.md in dist-info | PERMISSIVE |
| nodeenv | 1.12.0 | transitive, dev | BSD-3-Clause | License field 'BSD' + Classifier BSD License; LICENSE text is 3-clause style (names of contributors may not endorse) | PERMISSIVE |
| numba | 0.68.1 | direct, runtime | BSD-2-Clause | License field 'BSD' + Classifier BSD License; LICENSE text is 2-clause | PERMISSIVE |
| numpy | 2.5.4 | direct, runtime | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | License-Expression; LICENSE.txt in dist-info | PERMISSIVE |
| opentelemetry-api | 1.45.2 | transitive, runtime | Apache-2.0 | License-Expression; LICENSE in dist-info | PERMISSIVE |
| packaging | 26.3 | transitive, runtime (via streamlit and altair; pytest also needs it) | Apache-2.0 OR BSD-2-Clause | License-Expression; LICENSE.APACHE and LICENSE.BSD in dist-info | PERMISSIVE |
| pandas | 3.1.1 | transitive, runtime | BSD-3-Clause | License-Expression; LICENSE and LICENSES/ in dist-info | PERMISSIVE |
| patsy | 1.0.3 | transitive, runtime (arch via statsmodels) | BSD-2-Clause (bulk) and PSF-2.0 (portions) | License field '2-clause BSD' + Classifier BSD License; licenses/LICENSE.txt: "The bulk of Patsy is distributed under a simple 2-clause BSD license", plus a PSF License Version 2 section | PERMISSIVE |
| pillow | 13.0.0 | transitive, runtime | MIT-CMU | License-Expression; LICENSE in dist-info (HPND-style PIL licence) | PERMISSIVE |
| pluggy | 1.7.0 | transitive, dev | MIT | License field MIT + Classifier; LICENSE text confirms MIT | PERMISSIVE |
| polars | 2.1.0 | direct, runtime | MIT | Classifier MIT License; LICENSE text confirms MIT | PERMISSIVE |
| polars-runtime-32 | 2.1.0 | transitive, runtime | MIT | Classifier MIT License; LICENSE text confirms MIT | PERMISSIVE |
| protobuf | 7.36.2 | transitive, runtime | BSD-3-Clause | License field '3-Clause BSD License'; LICENSE in dist-info is 3-clause | PERMISSIVE |
| pyarrow | 26.0.1 | direct, runtime | Apache-2.0 | License-Expression; LICENSE.txt is Apache License 2.0 | PERMISSIVE |
| pycparser | 3.12 | transitive, runtime | BSD-3-Clause | License-Expression; LICENSE in dist-info | PERMISSIVE |
| pydantic | 2.14.3 | transitive, runtime | MIT | License-Expression; LICENSE in dist-info | PERMISSIVE |
| pydantic-core | 2.50.4 | transitive, runtime | MIT | License-Expression; LICENSE in dist-info | PERMISSIVE |
| pydeck | 0.9.3 | transitive, runtime | Apache-2.0 | License field 'Apache License 2.0'; LICENSE.txt in dist-info | PERMISSIVE |
| pygments | 2.21.0 | transitive, runtime | BSD-2-Clause | License-Expression; LICENSE in dist-info | PERMISSIVE |
| pyjwt | 2.15.1 | transitive, runtime | MIT | License-Expression; LICENSE in dist-info | PERMISSIVE |
| pyright | 1.1.414 | direct, dev | MIT | License field MIT; LICENSE text confirms MIT; bundled dist/LICENSE.txt is MIT | PERMISSIVE |
| pytest | 9.2.0 | direct, dev | MIT | License-Expression; LICENSE in dist-info | PERMISSIVE |
| python-dateutil | 4.0.0 | transitive, runtime | BSD-3-Clause | License-Expression; LICENSE in dist-info | PERMISSIVE |
| python-multipart | 0.0.34 | transitive, runtime | Apache-2.0 | License-Expression + Classifier Apache Software License; LICENSE.txt in dist-info | PERMISSIVE |
| pywin32 | 312 | transitive, runtime | PSF | License field PSF + Classifier PSF License in METADATA of the cp312 win_amd64 wheel at the lock URL (scratch fetch); component licence files PSF-style, BSD-style, MIT (MAPIStubLibrary) and LGPL-2.1 (adodbapi); Windows-only, not installed here | PERMISSIVE |
| referencing | 0.37.0 | transitive, runtime | MIT | License-Expression; COPYING in dist-info | PERMISSIVE |
| requests | 2.34.2 | transitive, runtime | Apache-2.0 | License field Apache-2.0 + Classifier; LICENSE and NOTICE in dist-info | PERMISSIVE |
| rich | 15.0.0 | transitive, runtime | MIT | License field MIT + Classifier; LICENSE text confirms MIT | PERMISSIVE |
| rpds-py | 2026.9.1 | transitive, runtime | MIT | License-Expression; LICENSE in dist-info | PERMISSIVE |
| ruff | 0.17.5 | direct, dev | MIT | License-Expression; LICENSE in dist-info | PERMISSIVE |
| scipy | 1.18.1 | transitive, runtime (arch, statsmodels) | BSD-3-Clause; bundled OpenBLAS/LAPACK BSD-3-Clause, GCC runtime GPL-3.0-or-later WITH GCC-exception-3.1, libquadmath LGPL-2.1-or-later | Classifier BSD License; LICENSE.txt in dist-info with Name/License lines per bundled library | FLAG |
| shellingham | 1.5.4 | transitive, runtime | ISC | License field 'ISC License' + Classifier ISCL; LICENSE text is ISC | PERMISSIVE |
| sse-starlette | 3.7.0 | transitive, runtime | BSD-3-Clause | License-Expression; LICENSE in dist-info | PERMISSIVE |
| starlette | 1.10.0 | transitive, runtime | BSD-3-Clause | License-Expression; LICENSE.md in dist-info | PERMISSIVE |
| statsmodels | 0.15.0 | transitive, runtime (arch) | BSD-3-Clause | License-Expression; licenses/LICENSE.txt in dist-info is 3-clause | PERMISSIVE |
| streamlit | 1.67.0 | direct, runtime | Apache-2.0 | License-Expression (the dist-info carries no LICENSE file) | PERMISSIVE |
| truststore | 0.10.5 | transitive, runtime | MIT | License-Expression; LICENSE in dist-info | PERMISSIVE |
| typer | 0.29.1 | direct, runtime | MIT | License-Expression; LICENSE in dist-info | PERMISSIVE |
| typing-extensions | 4.16.0 | transitive, runtime | PSF-2.0 | License-Expression; LICENSE in dist-info | PERMISSIVE |
| typing-inspection | 0.4.5 | transitive, runtime | MIT | License-Expression; LICENSE in dist-info | PERMISSIVE |
| tzdata | 2026.6 | transitive, runtime | Apache-2.0 | License field Apache-2.0; LICENSE and licenses/LICENSE_APACHE in dist-info | PERMISSIVE |
| urllib3 | 2.8.0 | transitive, runtime | MIT | License-Expression; LICENSE.txt in dist-info | PERMISSIVE |
| uvicorn | 0.54.0 | transitive, runtime | BSD-3-Clause | License-Expression; LICENSE.md in dist-info | PERMISSIVE |
| watchdog | 7.1.0 | transitive, runtime | Apache-2.0 | License field Apache-2.0; LICENSE, COPYING and AUTHORS in dist-info | PERMISSIVE |
| websockets | 17.4 | transitive, runtime | BSD-3-Clause | License-Expression; LICENSE in dist-info | PERMISSIVE |
| wrapt | 2.5.2 | transitive, runtime (arch via statsmodels, formulaic) | BSD-2-Clause | License-Expression; licenses/LICENSE in dist-info | PERMISSIVE |
