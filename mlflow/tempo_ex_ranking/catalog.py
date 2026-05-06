"""PyIceberg catalog config for the Tempo ranking project.

Local dev uses a SQLite-backed catalog and a file:// warehouse, so no
JVM, no Spark, no S3 required. Same code points at Glue / Polaris /
Nessie in prod by setting ICEBERG_CATALOG_URI and ICEBERG_WAREHOUSE.
"""
import os

from pyiceberg.catalog import Catalog, load_catalog
from pyiceberg.catalog.sql import SqlCatalog

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_WAREHOUSE = os.path.join(_HERE, 'warehouse')

NAMESPACE = 'tempo'
USERS_TABLE = f'{NAMESPACE}.users'
EXERCISES_TABLE = f'{NAMESPACE}.exercises'
EVENTS_TABLE = f'{NAMESPACE}.events'


def get_catalog() -> Catalog:
    """Return the configured catalog. SQLite-backed locally."""
    catalog_uri = os.environ.get('ICEBERG_CATALOG_URI')
    warehouse = os.environ.get('ICEBERG_WAREHOUSE', _DEFAULT_WAREHOUSE)

    # Prod path: a real catalog (Glue, Polaris, Nessie) loaded from env
    if catalog_uri and catalog_uri.startswith('rest://'):
        return load_catalog(
            'tempo',
            **{
                'type': 'rest',
                'uri': catalog_uri.replace('rest://', ''),
                'warehouse': warehouse,
            },
        )

    # Local path: SQLite catalog + filesystem warehouse
    os.makedirs(warehouse, exist_ok=True)
    db_path = os.path.join(warehouse, 'catalog.db')
    return SqlCatalog(
        'tempo',
        **{
            'uri': f'sqlite:///{db_path}',
            'warehouse': f'file://{warehouse}',
        },
    )


def ensure_namespace(catalog: Catalog, namespace: str = NAMESPACE) -> None:
    if (namespace,) not in catalog.list_namespaces():
        catalog.create_namespace(namespace)
