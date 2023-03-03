Contributing to wordview
========================

Thank you for contributing to wordview! We and the users of this repo
appreciate your efforts! If you spot a problem or you have a feature request
or you wanted to suggest an improvement, please create an issue. Please
first search the existing open and closed issues
`here <https://github.com/meghdadFar/wordview/issues>`__. If a related
issue already exists, you can add your comment and avoid creating
duplicate or very similar issues. If you come across an issue that you
would like to work on, feel free to `open a PR <#pull-request-pr>`__ for
it.

Branches
--------

To begin contributing, clone the repository and make sure you are on
``main`` branch. Then create your own branch.

.. code:: bash

   # Clone the repo
   git clone git@github.com:meghdadFar/wordview.git

   # Get the latest updates, if you have previously cloned wordview.
   git pull

   # Create a new branch
   git checkout -b <branch_name>

Please try to name your branch such that the name clarifies the purpose
of your branch, to some extent. We commonly use hyphenated branch names.
For instance, if you are developing an anomaly detection functionality
based on a normal distribution, a good branch name can be
``normal-dist-anomaly-detection``.

Environment Setup
-----------------

We use `Poetry <https://pypi.org/project/poetry/>`__ to manage
dependencies and packaging. Follow these steps to set up your dev
environment:

.. code:: bash

   python -m venv .venv

   source .venv/bin/activate

   pip install poetry

   # Disable Poetry's environment creation, since we already have created one
   poetry config virtualenvs.create false

Use Poetry to install dev (and main) dependencies:

.. code:: bash

   poetry install

By default, dependencies across all non-optional groups are install. See
`Poetry
documentation <https://python-poetry.org/docs/managing-dependencies/>`__
for more details and for instructions on how to define optional
dependency groups.

Testing
-------

Testing of ``wordview``\ ’s is carried out via
`Pytest <https://docs.pytest.org/>`__. Please include tests for any
piece of code that you create inside the `tests <./tests/>`__ directory.
To see examples, you can consult the existing tests in this directory.
Once you have provided the tests, simply run in the command line.

.. code:: bash

   python -m pytest tests

If all tests pass, you can continue with the next steps.

Code Quality
------------

To ensure a high quality in terms of readability, complying with PEP
standards, and static type checking, we use ``pre-commit`` with
``black``, ``flake8``, ``mypy`` and ``isort``. The configurations are in
``.pre-commit-config.yaml``. Once you have installed dev dependencies,
following the above instructions, run ``pre-commit install`` so that the
above tools are installed.

When ``pre-commit`` install its dependencies successfully, it runs
``black``, ``flake8``, ``mypy`` and ``isort`` each time you try to
commit code. If one of these tools fail, fix the issue, run
``git add <changed_file>`` again, and then again
``git commit -m <commit_message>``. Once you successfully committed your
changes, you can push your branch to remote and create a PR. You can now merge your PR, following
the instructions in `Pull Request (PR) <#pull-request-pr>`__. Note that you can
skip pre-commit checks by running your ``git commit`` with the ``--no-verify`` flag (e.g. ``git commit -m 'dirty fix' --no-verify``), however,
this is discouraged unless you really have to. 

Pull Request (PR)
-----------------

Once your work is complete, you can make a pull request. Remember to
link your pull request to an issue by using a supported keyword in the
pull request’s description or in a commit message. E.g. “closes
#issue_number”, “resolves #issue_number”, or “fixes #issue_number”. See
`this
page <https://docs.github.com/en/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue>`__
for more details.

Once your PR is submitted, a maintainer will review your PR. They may
ask questions or suggest changes either using `suggested
changes <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/reviewing-changes-in-pull-requests/incorporating-feedback-in-your-pull-request>`__
or pull request comments.

Once all the comments and changes are resolved, your PR will be merged.
🥳🥳

Thank you for your contribution! We are really excited to have your work
integrated in wordview!
