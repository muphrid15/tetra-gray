#include "operator.h"
#include "identity.h"
#include "procedure.h"
#include "list.h"
#include <string>
#include <iostream>

int main(void);
int curry_test(void);
int reverse_curry_test(void);
int compose_test(void);
int identity_unit_test(void);
int identity_bind_test(void);
int identity_global_join_test(void);
int identity_apply_test(void);
int identity_then_test(void);
ftk::Procedure<ftk::Empty> printLn(const std::string st);
int procedure_unit_test(void);
int procedure_fmap_test(void);
int procedure_join_test(void);
int list_concat_fold_test(void);
int list_range_map_test(void);
int list_unit_test(void);
int list_fmap_test(void);
int list_join_test(void);

int main(void)
{
	std::cout << "curry_test():  " << curry_test() << std::endl;
	std::cout << "reverse_curry_test(): " << reverse_curry_test() << std::endl;
	std::cout << "compose_test(): " << compose_test() << std::endl;
	std::cout << "identity_unit_test(): " << identity_unit_test() << std::endl;
	std::cout << "identity_bind_test(): " << identity_bind_test() << std::endl;
	std::cout << "identity_global_join_test(): " << identity_global_join_test() << std::endl;
	std::cout << "identity_apply_test(): " << identity_apply_test() << std::endl;
	std::cout << "identity_then_test(): " << identity_then_test() << std::endl;
	printLn(std::string("printLn() test: 3")).run();
	std::cout << "procedure_unit_test(): " << procedure_unit_test() << std::endl;
	std::cout << "procedure_fmap_test(): " << procedure_fmap_test() << std::endl;
	std::cout << procedure_join_test() << std::endl;
	std::cout << "list_concat_fold_test(): " << list_concat_fold_test() << std::endl;
	std::cout << "list_range_map_test(): " << list_range_map_test() << std::endl;
	std::cout << "list_unit_test(): " << list_unit_test() << std::endl;
	std::cout << "list_fmap_test(): " << list_fmap_test() << std::endl;
	std::cout << "list_join_test(): " << list_join_test() << std::endl;
	return 0;
}

int curry_test(void)
{
	using ftk::operator|;
	using ftk::operator>>;
	return 1 | [](int ix, int iy) { return ix * iy; } >> 3;
}

int reverse_curry_test(void)
{
	using ftk::operator|;
	using ftk::operator<<;
	return 6 | [](int ix, int iy) { return ix / iy; } << 2;
}

int compose_test(void)
{
	using ftk::operator|;
	using ftk::operator&;
	return 1 | ([](int ix) { return 2*ix; } & [](int iy) { return iy+1;});
}

int identity_unit_test(void)
{
	return 3 | ftk::Monad<ftk::Identity>::unit | ftk::runIdentity;
}

int identity_bind_test(void)
{
	return (6 | ftk::Monad<ftk::Identity>::unit) | (ftk::bind >> [](int ix) { return (ix/2) | ftk::Monad<ftk::Identity>::unit; } ) | ftk::runIdentity;
}

int identity_global_join_test(void)
{
	return 3 | ftk::Monad<ftk::Identity>::unit | ftk::Monad<ftk::Identity>::unit | ftk::join | ftk::runIdentity;
}

int identity_apply_test(void)
{
	return 1 | ftk::Applicative<ftk::Identity>::pure | ftk::apply >> ([](int ix) { return ix+2; } | ftk::Applicative<ftk::Identity>::pure) | ftk::runIdentity;
}

int identity_then_test(void)
{
	return 2 | ftk::Monad<ftk::Identity>::unit | ftk::then >> (3 | ftk::Monad<ftk::Identity>::unit) | ftk::runIdentity;
}

ftk::Procedure<ftk::Empty> printLn(const std::string st)
{
	return [st]() -> ftk::Empty { std::cout << st << std::endl; return ftk::Empty(); } | ftk::makeProcedure;
}

int procedure_unit_test(void)
{
	return (3 | ftk::Monad<ftk::Procedure>::unit).run();
}

int procedure_fmap_test(void)
{
	return (1 | ftk::Monad<ftk::Procedure>::unit | ftk::fmap >> [](int ix) { return ix + 2; }).run();
}

int procedure_join_test(void)
{
	return ([]() -> ftk::Procedure<int>
	{
		std::cout << "procedure_join_test";
		return []() -> int
		{
			std::cout << "(): ";
			return 3;
	   	} | ftk::makeProcedure;
	} | ftk::makeProcedure | ftk::join).run();
}

int list_concat_fold_test(void)
{
	//tests the constructor, concat, and fold
	return ftk::List<int>(2) | ftk::concat >> ftk::List<int>(1) | ftk::fold >> [](int ix, int iy) { return ix + iy; } >> 0;
}

int list_range_map_test(void)
{
	return ftk::ListLikeData<ftk::List<int> >::map([](int ix) { return ix-1; }, 3 | ftk::ListLike<ftk::List>::range >> 1) | ftk::fold >> [](int ix, int iy) { return ix+iy; } >> 0;
}

int list_unit_test(void)
{
	//similar to list_concat_fold
	return 2 | ftk::Monad<ftk::List>::unit | ftk::concat >> (1 | ftk::Monad<ftk::List>::unit) | ftk::fold >> [](int ix, int iy) { return ix+iy; } >> 0;
}

int list_fmap_test(void)
{
	return 3 | ftk::ListLike<ftk::List>::range >> 1 | ftk::fmap >> [](int ix) { return ix-1; } | ftk::fold >> [](int ix, int iy) { return ix + iy; } >> 0;
}

int list_join_test(void)
{
	using ftk::operator|;
	return 3 | ftk::ListLike<ftk::List>::range >> 1 | ftk::Monad<ftk::List>::unit | ftk::concat >> (3 | ftk::ListLike<ftk::List>::range >> 4 | ftk::Monad<ftk::List>::unit) | ftk::join | ftk::fold >> [](int ix, int iy) { return ix + iy; } >> 0 | [](int ix) { return ix/7; };
}
