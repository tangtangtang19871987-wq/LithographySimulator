## About
LithographySimulator is an open-source tool/toy for modeling optical lithography. 

Currently, it simulates partial coherence imaging with the Abbe formulation, Fraunhofer mask diffraction with binary masks, support for annular, quasar, and classical light sources, and arbitrary aberration modeling.

Depending on device support it uses PyTorch for GPU or CPU acceleration and can be reasonably agile when computing the aerial image. The main limiting factor will be VRAM or system RAM, as the current approach is very memory intensive.

![image](https://github.com/user-attachments/assets/bd68ebfc-20ad-4fec-95bf-31748b02c3e5)

![imgtt](https://github.com/user-attachments/assets/926141f3-e007-49c6-bcbc-21bd08f8942b)


## Goals
Right now, although it mostly works, a lot could still be done to improve it. Expect that much of this is incomplete in perpetuity, however, I do look over the code from time to time. Never say never!

- [x] Refactor architecture to be more usable (Objects, perhaps, rather than the current approach with haphazard application of global variables)
- [x] Add support for Zernike polynomial modeling of optical wavefront error for the pupil function. Currently, only defocus is supported
- [ ] Validate the correctness of the lithography model, either by testing against known-correct models or through formally validating the mathematics inside the program.
- [x] Add FFT approximation as appears in [1] alongside the classical solver
- [ ] Add GDSII/OASIS import
- [ ] Add photoresist response modeling, simple or otherwise
- [ ] 2D solver for lithography recipe generation
- [x] Allow for more complicated light sources like quasar or quadrupole

## Copilot CLI dashboard

The repository also contains a small, standard-library-only desktop dashboard
for supervising multiple Copilot (or other) command-line processes:

```bash
python copilot_dashboard.py
```

Each session has its own command, working directory, environment variables,
terminal output, input box, lifecycle status, progress indicator, and lifecycle
controls. Configurations can be saved as reusable profiles (under
`~/.config/copilot-dashboard/profiles.json`), while complete session logs can be
exported on demand. On POSIX systems the child receives a pseudo-terminal, so
interactive CLIs retain their normal terminal behaviour. Environment values are
supplied one `KEY=VALUE` per line; they are applied only to that session and do
not modify the dashboard process.

Additional reliability features include bounded on-screen log retention,
incremental UTF-8 decoding, atomic profile writes, command/directory validation,
explicit process states, non-zero exit highlighting, and graceful termination
with automatic forced-kill fallback. The dashboard never invokes commands via a
shell, which avoids unintended shell expansion of configuration values.

### Design research

The feature selection follows useful patterns from established open-source
terminal and process-management projects:

* [Process Compose](https://github.com/F1bonacc1/process-compose) demonstrates
  the value of per-process lifecycle state, restart controls, log visibility,
  and reusable configuration.
* [Wave Terminal](https://github.com/wavetermdev/waveterm) and
  [Tabby](https://github.com/Eugeny/tabby) demonstrate session-oriented tabs,
  persistent connection profiles, and immediate interactive terminal I/O.
* [ttyd](https://github.com/tsl0922/ttyd) demonstrates why interactive programs
  need a pseudo-terminal rather than ordinary stdout pipes.
* [pyte](https://github.com/selectel/pyte) is a useful future option if full
  terminal emulation (cursor movement, alternate screen, and colours) becomes
  more important than keeping this MVP standard-library-only.

The current UI intentionally stops short of terminal emulation: it presents a
safe, readable text transcript and strips ANSI control sequences. Secrets are
also stored as plain text when a profile is saved (with owner-only permissions
on POSIX), so tokens should preferably be inherited from the parent environment
or a platform credential manager.

## Acknowledgment and Citations
1. T.-S. Gau et al., “Ultra-fast aerial image simulation algorithm using wavelength scaling and fast Fourier transformation to speed up calculation by more than three orders of magnitude,” JM3 22(2), 023201, SPIE (2023) [doi:10.1117/1.JMM.22.2.023201].

Note: It is very important to note that the prior paper, Gao 2023, provided the starting code for this project. The original MATLAB code is available on request from the corresponding author. I translated the code into Python in a sensible manner and improved performance, but the physics underlying this model is owed in large, but not complete, part to this paper's code.

2. B. J. Lin, Optical Lithography: Here is why, SPIE (2021).
3. X. Wu et al., “Efficient source mask optimization with Zernike polynomial functions for source representation,” Opt. Express, OE 22(4), 3924–3937, Optica Publishing Group (2014) [doi:10.1364/OE.22.003924].
4. N. B. Cobb, “Fast optical and process proximity correction algorithms for integrated circuit manufacturing,” PhD, University of California, Berkeley (1998).
5. M. Guthaus, “mguthaus/DimmiLitho,” (2021).
6. P. Evanschitzky, A. Erdmann, and T. Fuehner, “Extended Abbe approach for fast and accurate lithography imaging simulations,” in 25th European Mask and Lithography Conference, pp. 1–11 (2009) [doi:10.1117/12.835168].
7. E. Hecht, Optics, Pearson Education, Incorporated (2017).
8. C. Mack, Fundamental Principles of Optical Lithography: The Science of Microfabrication, John Wiley & Sons (2008).
