##############################################################################
# Sequence
##############################################################################
import typing
import itertools
from py_trees import behaviour, common
from py_trees.composites import Sequence,Composite

class SequenceNoMem(Composite):
    """
    Sequences are the factory lines of Behaviour Trees

    .. graphviz:: dot/sequence.dot

    A sequence will progressively tick over each of its children so long as
    each child returns :data:`~py_trees.common.Status.SUCCESS`. If any child returns
    :data:`~py_trees.common.Status.FAILURE` or :data:`~py_trees.common.Status.RUNNING` the sequence will halt and the parent will adopt
    the result of this child. If it reaches the last child, it returns with
    that result regardless.

    .. note::

       The sequence halts once it sees a child is RUNNING and then returns
       the result. *It does not get stuck in the running behaviour*.

    .. seealso:: The :ref:`py-trees-demo-sequence-program` program demos a simple sequence in action.

    Args:
        name (:obj:`str`): the composite behaviour name
        children ([:class:`~py_trees.behaviour.Behaviour`]): list of children to add
        *args: variable length argument list
        **kwargs: arbitrary keyword arguments

    """
    def __init__(self, name="SequenceNoMem", children=None, *args, **kwargs):
        super(SequenceNoMem, self).__init__(name, children, *args, **kwargs)
        self.current_index = -1  # -1 indicates uninitialised
        self.memory = False
        self.current_child = None

    def tick(self):
        """
        Tick over the children.

        Yields:
            :class:`~py_trees.behaviour.Behaviour`: a reference to itself or one of its children
        """
        self.logger.debug("%s.tick()" % self.__class__.__name__)

        #initialize
        index = 0
        if self.status != common.Status.RUNNING:
            self.current_child = self.children[0] if self.children else None
            for child in self.children:
                # reset the children, this helps when introspecting the tree
                if child.status != common.Status.INVALID:
                    child.stop(common.Status.INVALID)
            # subclass (user) handling
            self.initialise()
        elif self.memory and common.Status.RUNNING:
            assert self.current_child is not None  # should never be true, help mypy out
            index = self.children.index(self.current_child)
        elif not self.memory and common.Status.RUNNING:
            self.current_child = self.children[0] if self.children else None
        else:
            # previous conditional checks should cover all variations
            raise RuntimeError("Sequence reached an unknown / invalid state")
        
        # nothing to do
        if not self.children:
            self.current_child = None
            self.stop(common.Status.SUCCESS)
            yield self
            return
        
        for child in itertools.islice(self.children, index, None):
            for node in child.tick():
                yield node
                if node is child and node.status != common.Status.SUCCESS:
                    self.status = node.status
                    if not self.memory:
                        # invalidate the remainder of the sequence
                        # i.e. kill dangling runners
                        for child in itertools.islice(self.children, index + 1, None):
                            if child.status != common.Status.INVALID:
                                child.stop(common.Status.INVALID)
                    yield self
                    return
            try:
                # advance if there is 'next' sibling
                self.current_child = self.children[index + 1]
                index += 1
            except IndexError:
                pass
        # At this point, all children are happy with their SUCCESS, so we should be happy too
        self.stop(common.Status.SUCCESS)
        yield self

    # @property
    # def current_child(self):
    #     """
    #     Have to check if there's anything actually running first.

    #     Returns:
    #         :class:`~py_trees.behaviour.Behaviour`: the child that is currently running, or None
    #     """
    #     if self.current_index == -1:
    #         return None
    #     return self.children[self.current_index] if self.children else None
    # @current_child.setter
    # def current_child(self,val):
    #     self.current_child = val

    def stop(self, new_status=common.Status.INVALID):
        """
        Stopping a sequence requires taking care of the current index. Note that
        is important to implement this here intead of terminate, so users are free
        to subclass this easily with their own terminate and not have to remember
        that they need to call this function manually.

        Args:
            new_status (:class:`~py_trees.common.Status`): the composite is transitioning to this new status
        """
        # retain information about the last running child if the new status is
        # SUCCESS or FAILURE
        Composite.stop(self, new_status)


class MySequenceNoMem(Sequence):
    """
    Sequences are the factory lines of behaviour trees.

    .. graphviz:: dot/sequence.dot

    A sequence will progressively tick over each of its children so long as
    each child returns :data:`~py_trees.common.Status.SUCCESS`. If any child returns
    :data:`~py_trees.common.Status.FAILURE` or :data:`~py_trees.common.Status.RUNNING`
    the sequence will halt and the parent will adopt
    the result of this child. If it reaches the last child, it returns with
    that result regardless.

    .. note::

       The sequence halts once it engages with a child is RUNNING, remaining behaviours
       are not ticked.

    .. note::

       If configured with `memory` and a child returned with running on the previous tick, it will
       proceed directly to the running behaviour, skipping any and all preceding behaviours. With memory
       is useful for moving through a long running series of tasks. Without memory is useful if you
       want conditional guards in place preceding the work that you always want checked off.

    .. seealso:: The :ref:`py-trees-demo-sequence-program` program demos a simple sequence in action.

    Args:
        name: the composite behaviour name
        memory: if :data:`~py_trees.common.Status.RUNNING` on the previous tick,
            resume with the :data:`~py_trees.common.Status.RUNNING` child
        children: list of children to add
    """

    def __init__(
        self,
        name: str,
        memory: bool,
        children: typing.Optional[typing.List[behaviour.Behaviour]] = None,
    ):
        super(Sequence, self).__init__(name, children)
        self.memory = memory
        self.current_index = -1  # -1 indicates uninitialised


    def tick(self) -> typing.Iterator[behaviour.Behaviour]:
        """
        Tick over the children.

        Yields:
            :class:`~py_trees.behaviour.Behaviour`: a reference to itself or one of its children
        """
        self.logger.debug("%s.tick()" % self.__class__.__name__)

        # initialise
        index = 0
        if self.status != common.Status.RUNNING:
            self.current_child = self.children[0] if self.children else None
            for child in self.children:
                if child.status != common.Status.INVALID:
                    child.stop(common.Status.INVALID)
            self.initialise()  # user specific initialisation
        elif self.memory and common.Status.RUNNING:
            assert self.current_child is not None  # should never be true, help mypy out
            index = self.children.index(self.current_child)
        elif not self.memory and common.Status.RUNNING:
            self.current_child = self.children[0] if self.children else None
        else:
            # previous conditional checks should cover all variations
            raise RuntimeError("Sequence reached an unknown / invalid state")

        # nothing to do
        if not self.children:
            self.current_child = None
            self.stop(common.Status.SUCCESS)
            yield self
            return

        # actual work
        for child in itertools.islice(self.children, index, None):
            for node in child.tick():
                yield node
                if node is child and node.status != common.Status.SUCCESS:
                    self.status = node.status
                    if not self.memory:
                        # invalidate the remainder of the sequence
                        # i.e. kill dangling runners
                        for child in itertools.islice(self.children, index + 1, None):
                            if child.status != common.Status.INVALID:
                                child.stop(common.Status.INVALID)
                    yield self
                    return
            try:
                # advance if there is 'next' sibling
                self.current_child = self.children[index + 1]
                index += 1
            except IndexError:
                pass

        self.stop(common.Status.SUCCESS)
        yield self

    def stop(self, new_status: common.Status = common.Status.INVALID) -> None:
        """
        Ensure that children are appropriately stopped and update status.

        Args:
            new_status : the composite is transitioning to this new status
        """
        self.logger.debug(
            f"{self.__class__.__name__}.stop()[{self.status}->{new_status}]"
        )
        Composite.stop(self, new_status)


