from pkg_resources import DistributionNotFound, get_distribution

try:
    version = get_distribution("sunkit_image").version
except DistributionNotFound:
    version = "unknown.dev"
