"""
aliases.py — Domain table and column synonym mappings for schema embedding enrichment.
"""

TABLE_ALIASES = {
    "ITEM": ["item master", "part", "parts", "product", "products", "sku master", "material", "materials"],
    "SKUITEM": ["sku", "skus", "unit", "units", "barcode", "barcodes", "serial", "serials", "batch", "batches"],
    "SULOCATION": ["stock", "inventory", "su location", "storage location", "on hand", "quantity on hand", "suid", "storage unit"],
    "LOCATION": ["bin", "bins", "rack", "racks", "aisle", "aisles", "bay", "bays", "shelf", "shelves", "location master"],
    "PICKLIST": ["pick list", "picking task", "pick order", "fulfillment order", "picking assignment", "picklist status"],
    "PICKLISTITEM": ["picklist line", "picklist items", "pick list detail", "items to pick", "picklist qty"],
    "PICKLISTVIEW": ["picklist summary", "picklist details view", "picklist list view"],
    "GRN": ["goods receipt note", "goods receipt", "grn number", "receipt", "received items", "inward", "inbound", "vendor receipt"],
    "FGMODEL": ["finished goods model", "fg model", "model master", "fg code"],
    "ITEMLOCACNMAP": ["item location mapping", "warehouse item map", "location mapping"],
    "FGTRANSACTION": ["finished goods transaction", "fg transaction", "vin transaction", "vehicle transaction", "putaway transaction"],
    "SUIDACTIVITYLOG": ["suid log", "activity log", "suid movement", "suid history"],
    "WAREHOUSE": ["warehouse master", "plant", "site", "depot", "facility"],
    "user": ["user master", "system user", "operator", "picker", "warehouse user", "assigned user"],
}

COLUMN_ALIASES = {
    "ITEM": {
        "itemCode": ["part number", "part code", "material number", "product code"],
        "itemDescription": ["part name", "material description", "product description", "item name"],
    },
    "SKUITEM": {
        "skuCode": ["barcode", "serial number", "sku number"],
        "suid": ["suid code", "storage unit id"],
    },
    "SULOCATION": {
        "qty": ["quantity", "stock count", "inventory count", "available stock", "on hand stock"],
    },
    "LOCATION": {
        "locationCode": ["bin number", "rack code", "bin name", "location id string"],
    },
    "PICKLIST": {
        "picklistCode": ["pick list number", "pick task id", "pick order code"],
        "status": ["picklist state", "picking status", "completion status"],
        "assignedUser": ["picker name", "operator assigned"],
    },
    "GRN": {
        "grnNumber": ["grn code", "receipt number", "inward number"],
        "vendorCode": ["supplier code", "vendor id"],
        "vendorName": ["supplier name", "vendor title"],
    },
    "FGTRANSACTION": {
        "vin": ["vin number", "chassis number", "vehicle id"],
        "fgCode": ["finished goods code", "fg number"],
    },
}
