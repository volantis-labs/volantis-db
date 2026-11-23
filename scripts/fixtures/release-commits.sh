# Commits of the historical releases whose on-disk formats the compat
# fixtures freeze. These trees predate the rename and still use the old
# crate names, so the builders below address them by those names.
release_commit() {
  case $1 in
    v0.4.2) echo 37696c1146c1cb37bb6d224201f5e283cd0ca905 ;;
    v0.5.0) echo c77e74f1f1ffc07c442ace5a70ce444c43c921ff ;;
    v0.6.0) echo cae370f0615d5ddc05a5f46019d34c107d464d30 ;;
    *) echo "unknown release $1" >&2; return 1 ;;
  esac
}
