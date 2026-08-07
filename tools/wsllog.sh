#!/bin/sh
# Print the first lines of the last vvcudaemu build log.
head -"${1:-30}" /tmp/vvbuild.log
