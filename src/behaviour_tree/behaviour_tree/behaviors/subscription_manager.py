#!/usr/bin/env python3
"""
Subscription management utilities for behavior tree nodes.
Provides centralized subscription lifecycle management.
"""

def create_managed_subscription(node, msg_type, topic, callback, qos_profile=10, callback_group_type='sensing'):
    """
    Create a managed subscription that handles lifecycle and prevents duplicates.
    
    Args:
        node: ROS node with subscription_registry and callback_group_pool
        msg_type: Message type class
        topic: Topic name string
        callback: Callback function
        qos_profile: QoS profile (default 10)
        callback_group_type: Type of callback group ('control', 'sensing', 'coordination')
    
    Returns:
        subscription object or existing subscription if already exists
    """
    registry = node.subscription_registry
    pool = node.callback_group_pool
    
    # Check if subscription already exists
    if topic in registry['active_subscriptions']:
        # Increment reference count
        registry['subscription_counts'][topic] = registry['subscription_counts'].get(topic, 0) + 1
        print(f"[SUBSCRIPTION] Reusing existing subscription for {topic} (refs: {registry['subscription_counts'][topic]})")
        return registry['active_subscriptions'][topic]
    
    # Create new subscription with appropriate callback group
    callback_group = pool.get(callback_group_type, pool['sensing'])  # Default to sensing
    
    try:
        subscription = node.create_subscription(
            msg_type, topic, callback, qos_profile, callback_group=callback_group
        )
        
        # Register subscription
        registry['active_subscriptions'][topic] = subscription
        registry['subscription_counts'][topic] = 1
        
        print(f"[SUBSCRIPTION] Created new managed subscription for {topic} with {callback_group_type} callback group")
        return subscription
        
    except Exception as e:
        print(f"[SUBSCRIPTION] ERROR: Failed to create subscription for {topic}: {e}")
        return None


def destroy_managed_subscription(node, topic):
    """
    Destroy a managed subscription with reference counting.
    
    Args:
        node: ROS node with subscription_registry
        topic: Topic name string
    
    Returns:
        True if subscription was destroyed, False if still has references
    """
    registry = node.subscription_registry
    
    if topic not in registry['active_subscriptions']:
        print(f"[SUBSCRIPTION] WARNING: Attempted to destroy non-existent subscription: {topic}")
        return True
    
    # Decrement reference count
    registry['subscription_counts'][topic] -= 1
    
    # Only destroy if no more references
    if registry['subscription_counts'][topic] <= 0:
        try:
            subscription = registry['active_subscriptions'][topic]
            node.destroy_subscription(subscription)
            
            # Clean up registry
            del registry['active_subscriptions'][topic]
            del registry['subscription_counts'][topic]
            
            print(f"[SUBSCRIPTION] Destroyed managed subscription for {topic}")
            return True
            
        except Exception as e:
            print(f"[SUBSCRIPTION] ERROR: Failed to destroy subscription for {topic}: {e}")
            return False
    else:
        print(f"[SUBSCRIPTION] Kept subscription for {topic} (refs: {registry['subscription_counts'][topic]})")
        return False


def cleanup_all_managed_subscriptions(node):
    """
    Clean up all managed subscriptions for a node.
    
    Args:
        node: ROS node with subscription_registry
    """
    registry = node.subscription_registry
    
    print(f"[SUBSCRIPTION] Cleaning up {len(registry['active_subscriptions'])} managed subscriptions...")
    
    # Destroy all active subscriptions
    for topic, subscription in list(registry['active_subscriptions'].items()):
        try:
            node.destroy_subscription(subscription)
            print(f"[SUBSCRIPTION] Cleaned up subscription: {topic}")
        except Exception as e:
            print(f"[SUBSCRIPTION] ERROR cleaning up {topic}: {e}")
    
    # Clear registry
    registry['active_subscriptions'].clear()
    registry['subscription_counts'].clear()
    
    # Execute cleanup callbacks
    for cleanup_callback in registry['cleanup_callbacks']:
        try:
            cleanup_callback()
        except Exception as e:
            print(f"[SUBSCRIPTION] ERROR in cleanup callback: {e}")
    
    registry['cleanup_callbacks'].clear()
    print(f"[SUBSCRIPTION] All managed subscriptions cleaned up")
