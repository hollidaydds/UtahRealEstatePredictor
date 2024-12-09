from predict_house_price import predict_house_price

# Test different counties to see the price differences
test_cases = [
    {
        'bedrooms': 4,
        'bathrooms': 2.5,
        'square_footage': 2500,
        'lot_size': 0.25,
        'year_built': 2000,
        'county': 'Salt Lake',
        'property_type': 'Single Family'
    },
    {
        'bedrooms': 4,
        'bathrooms': 2.5,
        'square_footage': 2500,
        'lot_size': 0.25,
        'year_built': 2000,
        'county': 'Utah',
        'property_type': 'Single Family'
    },
    {
        'bedrooms': 4,
        'bathrooms': 2.5,
        'square_footage': 2500,
        'lot_size': 0.25,
        'year_built': 2000,
        'county': 'Davis',
        'property_type': 'Single Family'
    }
]

# Test each case
for i, case in enumerate(test_cases, 1):
    try:
        price = predict_house_price(**case)
        print(f"\nTest Case {i} ({case['county']} County):")
        print(f"Predicted Price: ${price:,.2f}")
    except Exception as e:
        print(f"Error with test case {i}: {e}")