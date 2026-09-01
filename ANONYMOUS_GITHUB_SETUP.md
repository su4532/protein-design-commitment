# Anonymous repository and mirror setup

Use the verified archive as the only source of repository contents. Do not mirror or fork a working repository that contains personal history.

## 1. Create a neutral private repository

Create an empty private repository from the neutral review organization. Apply these settings before pushing:

- visibility: **Private**;
- initialize with README: **Off**;
- initialize with `.gitignore`: **Off**;
- initialize with license: **Off**;
- issues, projects, wiki, discussions, and sponsorships: **Off**;
- Actions: **Disabled** unless the anonymous organization provides a neutral runner;
- forking: **Disallowed**;
- web commit signing: **Off**;
- default branch after the first push: `main`.

Do not add a personal account as a visible collaborator. Grant access through a neutral review team whose membership is not public.

## 2. Create a fresh neutral history

Extract `anonymous-code.zip` into a new empty directory. Verify it before initializing Git:

```bash
python scripts/verify_anonymous_release.py
git init -b main
git config user.name "Anonymous Authors"
git config user.email "anonymous@users.noreply.github.com"
git add README.md ANONYMOUS_GITHUB_SETUP.md ANONYMITY_AUDIT.json \
  CHECKSUMS.sha256 CODE_COMPLETENESS_MANIFEST.json pyproject.toml requirements-cpu.txt \
  analysis scripts configs schemas fixtures tests generated_figures generated_tables
git commit -m "Initial anonymous code release"
```

Confirm that the commit has the neutral author and contains one fresh root commit:

```bash
git log --format=fuller --decorate --stat -1
git rev-list --max-parents=0 --all
```

The second command must print exactly one commit. Do not import tags, branches, pull-request references, notes, reflogs, or Git LFS metadata from another repository.

## 3. Push through a draft pull request

Add the empty private repository as `origin`, substituting the neutral organization and repository names locally:

```bash
git checkout -b release-review
git remote add origin <neutral-private-repository-url>
git push -u origin release-review
```

Open a draft pull request from `release-review` to `main`. Use a neutral title such as `Anonymous code release`. The pull-request body should contain only:

- the command `python scripts/verify_anonymous_release.py`;
- the verifier's final PASS line;
- the SHA-256 digest of `anonymous-code.zip` from the external release receipt.

Do not paste local paths, personal repository URLs, build logs, tracking links, run identifiers, or provenance timestamps.

After review, mark the pull request ready, squash-merge it, and delete `release-review`. Record the resulting `main` commit hash in the submission notes kept outside the repository.

## 4. Protect the reviewed commit

Apply a branch ruleset to `main`:

- restrict deletions and force pushes;
- require a pull request before merging;
- require one approval from the neutral review team;
- dismiss stale approvals;
- require all conversations to be resolved;
- block branch updates when requirements are not met;
- allow no bypass actors;
- require linear history;
- lock the branch after the reviewed commit is final.

Do not enable a status check that exposes a personal runner, organization, cache, or log URL.

## 5. Create the read-only anonymous mirror

In `anonymous.4open.science`, create a repository mirror that reads the neutral private repository at the reviewed `main` commit. Configure:

- source branch: `main`;
- source revision: the reviewed commit hash, not a moving branch head;
- update mode: manual;
- write access: disabled;
- issue and pull-request mirroring: disabled;
- release, tag, action, and artifact mirroring: disabled;
- submodules and Git LFS: disabled;
- commit-history redaction: enabled;
- displayed owner and contact fields: blank;
- mirror landing page: repository files only.

Apply this redaction list to filenames, file contents, Git metadata, and rendered pages:

- personal names, usernames, email addresses, affiliations, and ORCID identifiers;
- personal or organizational repository URLs except the pinned public model dependency;
- internal filesystem paths, cluster names, run identifiers, job identifiers, and tracking URLs;
- provenance timestamps, review correspondence, and approval records;
- references to non-study repositories or earlier project directions.

## 6. Signed-out check

Open the mirror in a signed-out browser and confirm:

1. the repository loads without authentication;
2. the displayed commit matches the reviewed commit hash;
3. no owner identity, collaborator, source URL, or personal avatar is visible;
4. all files can be downloaded;
5. `python scripts/verify_anonymous_release.py` passes on the downloaded archive;
6. the nine figures and eight tables render or download correctly;
7. repository search returns no redaction-list findings.

Only the `anonymous.4open.science` mirror URL should be added to the manuscript appendix or review submission. Keep the neutral private source URL outside the manuscript and submission fields.
